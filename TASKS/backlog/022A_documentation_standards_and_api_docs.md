# Task 022A: Documentation Standards & API Documentation

## Metadata
- **Status**: backlog
- **Priority**: medium
- **Complexity**: medium
- **Category**: documentation
- **Estimated Effort**: 4-6 hours
- **Dependencies**: None
- **Assigned To**: Unassigned
- **Created**: 2025-11-21
- **Parent Task**: 022 (split into 022A, 022B)

## Context

Project Sanctuary has extensive documentation but lacks standardization and automated API documentation. This task establishes standards and generates API docs.

## Objective

Create documentation standards, templates, and automated API documentation generation with Sphinx.

## Acceptance Criteria

### 1. Documentation Standards
- [ ] Create `docs/DOCUMENTATION_STANDARDS.md` defining:
  - Markdown formatting conventions
  - Protocol document structure
  - Code documentation requirements (docstring format)
  - Diagram and visualization standards
- [ ] Create documentation templates:
  - `docs/templates/protocol_template.md`
  - `docs/templates/module_readme_template.md`
  - `docs/templates/api_documentation_template.md`

### 2. API Documentation Setup
- [ ] Set up Sphinx in `docs/` directory
- [ ] Configure autodoc for automatic docstring extraction
- [ ] Configure Napoleon for Google/NumPy style docstrings
- [ ] Add sphinx-rtd-theme for clean presentation
- [ ] Create `docs/conf.py` with proper configuration

### 3. Docstring Coverage
- [ ] Add docstrings to all public functions in `mnemonic_cortex/core/`
- [ ] Add docstrings to all public functions in `council_orchestrator/orchestrator/`
- [ ] Add docstrings to key scripts in `forge/scripts/`
- [ ] Achieve 90%+ docstring coverage for public APIs
- [ ] Follow Google docstring format consistently

### 4. API Documentation Generation
- [ ] Create `tools/docs/generate_api_docs.py` script
- [ ] Generate API documentation for:
  - `mnemonic_cortex/` modules
  - `council_orchestrator/` modules
  - `forge/` scripts
- [ ] Build HTML documentation
- [ ] Add documentation build to CI/CD

### 5. Documentation Validation
- [ ] Create `tools/docs/validate_documentation.py` script
- [ ] Check for missing docstrings
- [ ] Validate docstring format
- [ ] Check for broken links
- [ ] Generate documentation quality report

## Technical Approach

```markdown
# docs/DOCUMENTATION_STANDARDS.md

## Docstring Format

All public functions must use Google-style docstrings:

\`\`\`python
def query_cortex(query: str, k: int = 5) -> List[Document]:
    """
    Query the Mnemonic Cortex for relevant documents.
    
    This function implements the Parent Document Retriever pattern,
    returning complete documents rather than fragmented chunks.
    
    Args:
        query: Natural language query string
        k: Number of results to return (default: 5)
    
    Returns:
        List of Document objects with content and metadata
    
    Raises:
        ValueError: If query is empty or k < 1
        ChromaDBError: If vector database is unavailable
    
    Example:
        >>> results = query_cortex("What is Protocol 78?", k=3)
        >>> print(results[0].content)
    
    See Also:
        - Protocol 85: The Mnemonic Cortex Protocol
        - mnemonic_cortex/RAG_STRATEGIES_AND_DOCTRINE.md
    """
    pass
\`\`\`
```

```python
# docs/conf.py
project = 'Project Sanctuary'
copyright = '2025, Project Sanctuary'
author = 'Project Sanctuary'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
}
```

```python
# tools/docs/generate_api_docs.py
"""Generate API documentation using Sphinx."""

import subprocess
from pathlib import Path

def generate_api_docs():
    """Generate API documentation."""
    docs_dir = Path("docs")
    
    # Run sphinx-apidoc
    subprocess.run([
        "sphinx-apidoc",
        "-f",  # Force overwrite
        "-o", str(docs_dir / "api"),
        "mnemonic_cortex",
        "council_orchestrator",
        "forge"
    ], check=True)
    
    # Build HTML docs
    subprocess.run([
        "sphinx-build",
        "-b", "html",
        str(docs_dir),
        str(docs_dir / "_build" / "html")
    ], check=True)
    
    print(f"API documentation generated in {docs_dir}/_build/html/")

if __name__ == "__main__":
    generate_api_docs()
```

## Success Metrics

- [ ] Documentation standards document complete
- [ ] 90%+ docstring coverage for public APIs
- [ ] API documentation builds without errors
- [ ] HTML documentation accessible and navigable
- [ ] Automated doc generation in CI/CD

## Related Protocols

- **P89**: The Clean Forge
- **P115**: The Tactical Mandate
