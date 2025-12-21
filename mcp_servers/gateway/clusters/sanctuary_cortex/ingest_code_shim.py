#!/usr/bin/env python3
"""
Code-to-Markdown Ingestion Shim

This script converts Python code files into markdown format optimized for
RAG Cortex ingestion. It uses Python's AST module to parse code structure
and extract metadata without requiring LLM tokens.

Strategy: AST-Based "Pseudo-Markdown" Conversion
- Zero tokens: Uses built-in ast module (millisecond parsing)
- No agents: Hard-coded logic, no orchestrator needed
- Fast: <100ms per file
- Existing infrastructure: Feeds into cortex_ingest_incremental

Task: 110 - Extend RAG Cortex to Ingest Code Files
Author: Antigravity AI (based on Gemini 3.0 Pro strategy)
Date: 2025-12-14
"""

import ast
import os
import sys
from pathlib import Path
from typing import Optional


def parse_python_to_markdown(file_path: str) -> str:
    """
    Reads a .py file and converts it into a Markdown string 
    optimized for RAG Cortex ingestion.
    
    Uses Python's AST module to:
    - Extract module-level docstrings
    - Identify functions and classes
    - Extract docstrings and signatures
    - Preserve source code with syntax highlighting
    
    Args:
        file_path: Path to the Python file to convert
        
    Returns:
        Markdown-formatted string ready for RAG ingestion
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        SyntaxError: If the Python file has syntax errors
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise SyntaxError(f"Failed to parse {file_path}: {e}")
    
    filename = file_path.name
    try:
        relative_path = file_path.relative_to(Path.cwd()) if file_path.is_absolute() else file_path
    except ValueError:
        relative_path = file_path
    
    # Header acts like a file summary
    markdown_output = f"# Code File: {filename}\n\n"
    markdown_output += f"**Path:** `{relative_path}`\n"
    markdown_output += f"**Language:** Python\n"
    markdown_output += f"**Type:** Code Implementation\n\n"
    
    # Extract Global Docstring if exists
    docstring = ast.get_docstring(tree)
    if docstring:
        markdown_output += f"## Module Description\n\n{docstring}\n\n"
    
    # Extract imports for context
    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)
    
    if imports:
        markdown_output += f"## Dependencies\n\n"
        for imp in imports[:10]:  # Limit to first 10 to avoid clutter
            markdown_output += f"- `{imp}`\n"
        if len(imports) > 10:
            markdown_output += f"- ... and {len(imports) - 10} more\n"
        markdown_output += "\n"
    
    # Iterate through functions and classes (The "Chunking Strategy")
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Extract Metadata
            name = node.name
            
            if isinstance(node, ast.ClassDef):
                type_label = "Class"
            elif isinstance(node, ast.AsyncFunctionDef):
                type_label = "Async Function"
            else:
                type_label = "Function"
            
            start_line = node.lineno
            
            # Get the raw source code for this specific function/class
            segment = ast.get_source_segment(source, node)
            
            if segment is None:
                # Fallback: extract manually from source lines
                end_line = getattr(node, 'end_lineno', start_line)
                source_lines = source.split('\n')
                segment = '\n'.join(source_lines[start_line-1:end_line])
            
            # Extract docstring for the function/class
            func_doc = ast.get_docstring(node) or "No documentation provided."
            
            # Extract function signature for functions
            signature = ""
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = []
                for arg in node.args.args:
                    arg_str = arg.arg
                    if arg.annotation:
                        # Try to get annotation as string
                        try:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        except:
                            pass
                    args.append(arg_str)
                
                # Add return type if available
                returns = ""
                if node.returns:
                    try:
                        returns = f" -> {ast.unparse(node.returns)}"
                    except:
                        pass
                
                signature = f"({', '.join(args)}){returns}"
            
            # Format as a Markdown Section (This is what RAG likes)
            markdown_output += f"## {type_label}: `{name}`\n\n"
            markdown_output += f"**Line:** {start_line}\n"
            if signature:
                markdown_output += f"**Signature:** `{name}{signature}`\n"
            markdown_output += f"\n**Documentation:**\n\n{func_doc}\n\n"
            markdown_output += f"**Source Code:**\n\n```python\n{segment}\n```\n\n"
            
            # For classes, also extract methods
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                if methods:
                    markdown_output += f"**Methods:** {', '.join([f'`{m.name}`' for m in methods])}\n\n"
    
    # Add footer with metadata
    markdown_output += "---\n\n"
    markdown_output += f"**Generated by:** Code Ingestion Shim (Task 110)\n"
    markdown_output += f"**Source File:** `{relative_path}`\n"
    markdown_output += f"**Total Lines:** {len(source.split(chr(10)))}\n"
    
    return markdown_output


def parse_javascript_to_markdown(file_path: Path) -> str:
    """
    Reads a .js/.ts file and converts it into a Markdown string using Regex.
    
    Args:
        file_path: Path to the JS/TS file
        
    Returns:
        Markdown-formatted string
    """
    import re
    
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
        
    filename = file_path.name
    try:
        relative_path = file_path.relative_to(Path.cwd()) if file_path.is_absolute() else file_path
    except ValueError:
        relative_path = file_path
        
    # Header
    markdown_output = f"# Code File: {filename}\n\n"
    markdown_output += f"**Path:** `{relative_path}`\n"
    markdown_output += f"**Language:** JavaScript/TypeScript\n\n"
    
    # Simple formatting: Extract doc comments blocks (/** ... */)
    # This is a basic heuristic
    doc_blocks = re.finditer(r'/\*\*(.*?)\*/', source, re.DOTALL)
    for match in doc_blocks:
        comment = match.group(1).strip()
        # Clean up asterisks
        cleaned_comment = '\n'.join([line.strip().lstrip('*').strip() for line in comment.split('\n')])
        if len(cleaned_comment) > 20: # Arbitrary filter for significant comments
             markdown_output += f"## Documentation Hint\n\n{cleaned_comment}\n\n"

    # Identify Functions (Basic Regex)
    # 1. function foo()
    func_pattern = re.compile(r'function\s+(\w+)\s*\((.*?)\)')
    for match in func_pattern.finditer(source):
        name = match.group(1)
        args = match.group(2)
        line_no = source[:match.start()].count('\n') + 1
        markdown_output += f"## Function: `{name}`\n\n"
        markdown_output += f"**Line:** {line_no}\n"
        markdown_output += f"**Signature:** `function {name}({args})`\n\n"
        
        # Context extraction (dumb implementation: just grab next 10 lines)
        full_lines = source.split('\n')
        start_idx = max(0, line_no - 1)
        end_idx = min(len(full_lines), start_idx + 20) # Grab up to 20 lines
        snippet = '\n'.join(full_lines[start_idx:end_idx])
        markdown_output += f"```javascript\n{snippet}\n...\n```\n\n"

    # 2. const foo = () => 
    arrow_pattern = re.compile(r'(const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(?(.*?)\)?\s*=>')
    for match in arrow_pattern.finditer(source):
        kind = match.group(1)
        name = match.group(2)
        args = match.group(3) or ""
        line_no = source[:match.start()].count('\n') + 1
        markdown_output += f"## {kind.title()} Function: `{name}`\n\n"
        markdown_output += f"**Line:** {line_no}\n"
        markdown_output += f"**Signature:** `{name} = ({args}) => ...`\n\n"
        
        full_lines = source.split('\n')
        start_idx = max(0, line_no - 1)
        end_idx = min(len(full_lines), start_idx + 20)
        snippet = '\n'.join(full_lines[start_idx:end_idx])
        markdown_output += f"```javascript\n{snippet}\n...\n```\n\n"
            
    # Always include full source at bottom for reference if file is small (<500 lines)
    lines = source.split('\n')
    if len(lines) < 500:
         markdown_output += "## Full Source Code\n\n```javascript\n" + source + "\n```\n\n"
    
    return markdown_output


def convert_and_save(input_file: str, output_file: Optional[str] = None) -> str:
    """
    Convert a Python file to markdown and optionally save it.
    
    Args:
        input_file: Path to the Python file
        output_file: Optional path to save the markdown output
                    If None, uses input_file.md
                    
    Returns:
        Path to the output markdown file
    """
    input_path = Path(input_file)
    
    if output_file is None:
        # Append .md to the original filename (e.g., test.py -> test.py.md)
        # This prevents collisions with existing .md files and allows .gitignore filtering
        output_file = str(input_path) + ".md"
    
    output_path = Path(output_file)
    
    # Select parser based on extension
    if input_path.suffix == '.py':
        markdown_content = parse_python_to_markdown(input_path)
    elif input_path.suffix in ['.js', '.jsx', '.ts', '.tsx']:
        markdown_content = parse_javascript_to_markdown(input_path)
    else:
        # Fallback for unknown text files (treat as plain text block)
        with open(input_path, 'r', encoding='utf-8') as f:
             source = f.read()
        markdown_content = f"# File: {input_path.name}\n\n```\n{source}\n```"
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    return str(output_path)


def main():
    """CLI interface for the code ingestion shim."""
    if len(sys.argv) < 2:
        print("Usage: python ingest_code_shim.py <python_file> [output_file]")
        print("\nExample:")
        print("  python ingest_code_shim.py scripts/stabilizers/vector_consistency_check.py")
        print("  python ingest_code_shim.py my_code.py my_code_docs.md")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        output_path = convert_and_save(input_file, output_file)
        print(f"✅ Successfully converted {input_file}")
        print(f"📄 Output saved to: {output_path}")
        
        # Show stats
        with open(output_path, 'r') as f:
            content = f.read()
            lines = len(content.split('\n'))
            chars = len(content)
            print(f"📊 Stats: {lines} lines, {chars} characters")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
