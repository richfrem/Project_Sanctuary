import re
import datetime

# --- INSTRUCTIONS FOR GROUND CONTROL ---
# 1. Save your complete chat history with the Council into a plain text file (e.g., 'chat_log.txt').
# 2. Place this script in the same directory as your text file.
# 3. Run the script from your terminal: python compile_chronicle.py
# 4. A new file, 'LIVING_CHRONICLE.md', will be created with the formatted dialogue.
# 5. This is the file you will use, along with 'all_markdown_snapshot.txt', for the Prometheus Protocol.

def compile_chat_history(input_file='chat_log.txt', output_file='LIVING_CHRONICLE.md'):
    """
    Parses a raw chat log and converts it into a structured Markdown file
    for the Prometheus Protocol.
    """
    print(f"Reading from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_log = f.read()
    except FileNotFoundError:
        print(f"ERROR: Input file '{input_file}' not found. Please save your chat history.")
        return

    print(f"Compiling... Writing to {output_file}...")
    
    # Regex to find speaker lines (User, Gemini, Grok, GPT)
    speaker_pattern = re.compile(r'^(User|Gemini|Grok|GPT):', re.MULTILINE)
    
    # Split the log by speaker
    parts = speaker_pattern.split(raw_log)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# The Living Chronicle of the Sanctuary Council\n")
        f.write(f"## Compiled On: {datetime.datetime.utcnow().isoformat()}Z\n\n")
        f.write("---\n\n")

        # The first part is usually empty, so we start from the first speaker
        i = 1
        while i < len(parts):
            speaker = parts[i].strip()
            content = parts[i+1].strip()
            
            f.write(f"### Transmission from: **{speaker}**\n\n")
            f.write("```text\n")
            f.write(content)
            f.write("\n```\n\n")
            f.write("---\n\n")
            i += 2
            
    print(f"Compilation complete. '{output_file}' is ready.")

if __name__ == "__main__":
    compile_chat_history()