import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import NomicEmbeddings

# --- Constants ---
HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# --- Path Resolution ---
def find_project_root():
    """Find the project root by ascending from the current script's directory."""
    current_path = os.path.abspath(os.path.dirname(__file__))
    while True:
        # Project root is identified by the presence of the '.git' directory
        if '.git' in os.listdir(current_path):
            return current_path
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            raise FileNotFoundError("Could not find the project root (.git folder).")
        current_path = parent_path

def setup_environment(project_root):
    """Load environment variables from the .env file in the mnemonic_cortex directory."""
    dotenv_path = os.path.join(project_root, 'mnemonic_cortex', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        return True
    print(f"Warning: .env file not found at {dotenv_path}")
    return False

def main():
    """
    Main function to ingest the canonical cognitive genome into the vector database.
    """
    print("--- Starting Ingestion Process (Protocol 85, v1.2) ---")

    try:
        project_root = find_project_root()
        print(f"Project root identified at: {project_root}")

        if not setup_environment(project_root):
            print("Attempting to proceed with system environment variables.")

        db_path = os.getenv("DB_PATH")
        source_document_path = os.getenv("SOURCE_DOCUMENT_PATH")

        if not db_path or not source_document_path:
            raise ValueError("DB_PATH or SOURCE_DOCUMENT_PATH not set. Check your mnemonic_cortex/.env file.")

        # Construct full, absolute paths from the project root
        full_db_path = os.path.join(project_root, 'mnemonic_cortex', db_path)
        # The source document is now also relative to the project root
        full_source_path = os.path.join(project_root, source_document_path)

        print(f"Canonical source document path: {full_source_path}")
        print(f"Database will be persisted to: {full_db_path}")

        if not os.path.exists(full_source_path):
            raise FileNotFoundError(f"Canonical source document not found at '{full_source_path}'. Please run the 'capture_code_snapshot.js' script first.")

        # 1. Load, 2. Split, 3. Embed, 4. Store
        loader = TextLoader(full_source_path)
        documents = loader.load()
        print(f"Successfully loaded {len(documents)} document(s) from the canonical genome.")

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False)
        splits = markdown_splitter.split_text(documents[0].page_content)
        print(f"Split document into {len(splits)} chunks.")

        embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
        print("Initialized Nomic embedding model in local mode.")

        print("Creating and persisting vector store from canonical genome... (This may take a moment)")
        Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            persist_directory=full_db_path
        )
        print(f"Successfully created vector store at '{full_db_path}'.")
        print("--- Ingestion Process Complete ---")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n--- INGESTION FAILED ---")
        print(f"Error: {e}")
    except Exception as e:
        print(f"\n--- AN UNEXPECTED ERROR OCCURRED ---")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()