# mnemonic_cortex/app/services/vector_db_service.py
import os
import sys
import pickle
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_community.vectorstores import Chroma
from langchain_classic.storage import LocalFileStore, EncoderBackedStore # The Persistent Byte Store & Wrapper
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mnemonic_cortex.app.services.embedding_service import EmbeddingService

# --- CONFIGURATION: read from repo-root .env with sensible fallbacks ---
load_dotenv(dotenv_path=project_root / ".env")
DB_PATH = os.getenv("DB_PATH", "chroma_db")
# Use repo-root .env defaults so callers don't need hard-coded literals elsewhere
CHILD_COLLECTION = os.getenv("CHROMA_CHILD_COLLECTION", "child_chunks_v5")
PARENT_COLLECTION = os.getenv("CHROMA_PARENT_STORE", "parent_documents_v5")

_env = os.getenv("CHROMA_ROOT", "").strip()
# Prefer CHROMA_ROOT from .env (absolute or repo-relative); fall back to
# project layout (project_root / 'mnemonic_cortex' / DB_PATH) for backward compatibility.
CHROMA_ROOT = (Path(_env) if Path(_env).is_absolute() else (project_root / _env)).resolve() if _env else (project_root / "mnemonic_cortex" / DB_PATH)


def _detect_collections():
    """Return (child_collection_name, parent_collection_name) using env vars or auto-detection."""
    child = CHILD_COLLECTION or None
    parent = PARENT_COLLECTION or None
    try:
        if CHROMA_ROOT.exists() and CHROMA_ROOT.is_dir():
            for p in CHROMA_ROOT.iterdir():
                if not p.is_dir():
                    continue
                name = p.name
                if child is None and name.startswith("child_chunks"):
                    child = name
                if parent is None and name.startswith("parent_documents"):
                    parent = name
                if child and parent:
                    break
    except Exception:
        pass
    return child, parent


class VectorDBService:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorDBService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            print("[VectorDBService] Initializing with Ground Truth Architecture...")
            self.embedding_service = EmbeddingService()
            self.retriever = self._load_retriever()
            self.initialized = True

    def _load_retriever(self):
        # Resolve collection names and paths (env -> autodetect -> defaults)
        child_name, parent_name = _detect_collections()
        VECTORSTORE_PATH = os.path.join(str(CHROMA_ROOT), child_name)
        DOCSTORE_PATH = os.path.join(str(CHROMA_ROOT), parent_name)

        if not os.path.exists(VECTORSTORE_PATH) or not os.path.exists(DOCSTORE_PATH):
            raise FileNotFoundError(f"Required data stores not found at {VECTORSTORE_PATH} or {DOCSTORE_PATH}. Please run ingest.py.")

        vectorstore = Chroma(collection_name=child_name, persist_directory=VECTORSTORE_PATH, embedding_function=self.embedding_service.get_embedding_model())
        fs_store = LocalFileStore(root_path=DOCSTORE_PATH)
        # EncoderBackedStore constructor: (store, key_encoder, value_serializer, value_deserializer)
        store = EncoderBackedStore(
            store=fs_store,
            key_encoder=lambda k: str(k),
            value_serializer=pickle.dumps,
            value_deserializer=pickle.loads,
        )

        # Use a lightweight splitter consistent with ingestion
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        retriever = ParentDocumentRetriever(vectorstore=vectorstore, docstore=store, child_splitter=child_splitter)

        print("[VectorDBService] Retriever loaded successfully from persistent stores.")
        return retriever

    def query(self, text: str):
        print(f"[VectorDBService] Querying with text: '{text[:50]}...'")
        results = self.retriever.invoke(text)
        print(f"[VectorDBService] Found {len(results)} relevant parent documents.")
        return results

    # Compatibility wrapper: some callers (older main.py) expect a get_retriever() method.
    def get_retriever(self):
        """Return the internal retriever instance (backwards-compatible API)."""
        return self.retriever