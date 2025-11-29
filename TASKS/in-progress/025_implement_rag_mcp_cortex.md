# Task #025: Implement RAG MCP (Cortex)

**Status:** ~~Backlog~~ **SUPERSEDED BY TASK #050**  
**Priority:** ~~High~~ N/A  
**Lead:** Unassigned  
**Dependencies:** Task #028 (Pre-commit hooks), Shared Infrastructure  
**Domain:** `project_sanctuary.cognitive.cortex`  
**Related Documents:** `mnemonic_cortex/`, Model Context Protocol (MCP) specification

---

> [!IMPORTANT]
> **This task has been superseded by Task #050** which implemented a native MCP server approach instead of the FastAPI containerized approach described here. Task #050 is complete with all 4 tools operational, 28 unit tests + 3 integration tests passing, and full MCP integration.
> 
> This task is kept for reference as an alternative architecture approach.

## 1. Objective

Create a lightweight, containerized FastAPI microservice that wraps existing RAG logic from the `council_orchestrator` project and exposes it as a Model Context Protocol (MCP) Tool Server. This service will provide reliable, low-latency function calls for knowledge retrieval and incremental document ingestion, enabling self-hosted LLMs to access and expand their knowledge base through standardized API endpoints. The containerized approach ensures portability, isolation, and easy deployment via Podman.

## 2. Deliverables

### Core Application Files

1. **`mcp-rag-service/app/rag_engine.py`** - Core RAG engine implementing Parent Document Retriever pattern
2. **`mcp-rag-service/app/main.py`** - FastAPI application with `/rag/query` and `/rag/ingest` endpoints
3. **`mcp-rag-service/app/config.py`** - Configuration management using environment variables
4. **`mcp-rag-service/requirements.txt`** - Python dependencies matching existing Mnemonic Cortex stack

### Container Configuration Files

5. **`mcp-rag-service/Containerfile`** - Podman/Docker container definition (see below)
6. **`mcp-rag-service/docker-compose.yml`** - Optional compose file for easier deployment (see below)
7. **`mcp-rag-service/.env.example`** - Environment configuration template

### MCP Integration Files

8. **`mcp-rag-service/mcp_tools.yaml`** - MCP tool schema for LLM function calling (see below)
9. **`mcp-rag-service/examples/ollama_integration.py`** - Example Ollama integration script
10. **`mcp-rag-service/examples/test_endpoints.sh`** - Shell script for testing API endpoints

### Documentation Files

11. **`mcp-rag-service/README.md`** - Service-specific documentation
12. **Updated `mnemonic_cortex/VISION.md`** - Reflect Phase 1.5 completion
13. **Updated `mnemonic_cortex/EVOLUTION_PLAN_PHASES.md`** - Add Phase 1.5 section
14. **Updated `mnemonic_cortex/RAG_STRATEGIES_AND_DOCTRINE.md`** - Add incremental ingestion strategy
15. **Updated `mnemonic_cortex/README.md`** - Add MCP RAG usage instructions

---

## Complete Configuration Files

### `mcp-rag-service/Containerfile`

```dockerfile
# Containerfile - Production-ready Podman/Docker container for MCP RAG Service
# Based on Python 3.11 slim image for minimal footprint
FROM python:3.11-slim

# Metadata
LABEL maintainer="Project Sanctuary"
LABEL description="MCP RAG Tool Server - Containerized Mnemonic Cortex"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
# build-essential: Required for compiling Python packages with C extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy environment template
COPY .env.example .env

# Create directory for ChromaDB persistence
# This will be mounted as a volume in production
RUN mkdir -p /app/chroma_data && \
    chmod 755 /app/chroma_data

# Create non-root user for security
RUN useradd -m -u 1000 raguser && \
    chown -R raguser:raguser /app

# Switch to non-root user
USER raguser

# Expose API port
EXPOSE 8000

# Health check - verifies service is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI server with uvicorn
# --host 0.0.0.0: Listen on all interfaces (required for container networking)
# --port 8000: Default API port
# --workers 1: Single worker for development (increase for production)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### `mcp-rag-service/docker-compose.yml`

```yaml
# docker-compose.yml - Alternative deployment using Docker Compose
# Usage: docker-compose up -d
version: '3.8'

services:
  mcp-rag-service:
    build:
      context: .
      dockerfile: Containerfile
    container_name: mcp-rag-tool
    ports:
      - "8000:8000"
    volumes:
      # Persistent ChromaDB storage
      - ./data/chroma_data:/app/chroma_data
    environment:
      # Override defaults from .env
      - CHROMA_ROOT=/app/chroma_data
      - CHROMA_CHILD_COLLECTION=child_chunks_v5
      - CHROMA_PARENT_STORE=parent_documents_v5
      - API_HOST=0.0.0.0
      - API_PORT=8000
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s
    networks:
      - rag-network

networks:
  rag-network:
    driver: bridge
```

### `mcp-rag-service/mcp_tools.yaml`

```yaml
# mcp_tools.yaml - Model Context Protocol Tool Schema
# This file defines the RAG tools for LLM function calling
# Compatible with: Ollama, LangChain, OpenAI function calling, etc.

tools:
  - name: rag_query
    description: |
      Query the Mnemonic Cortex knowledge base for relevant context.
      
      Use this tool when you need to:
      - Answer questions about Project Sanctuary protocols, doctrines, or architecture
      - Retrieve canonical documentation or technical specifications
      - Access historical decisions, meeting notes, or chronicles
      - Verify information against the authoritative knowledge base
      
      The tool returns full parent documents (not fragments) to ensure complete context.
    
    parameters:
      type: object
      properties:
        question:
          type: string
          description: Natural language question to answer. Be specific and clear.
          examples:
            - "What is Protocol 87 and how does it work?"
            - "Summarize the Doctrine of Hybrid Cognition"
            - "What are the phases of the Mnemonic Cortex evolution plan?"
        
        top_k:
          type: integer
          description: Number of documents to retrieve (1-20). Default is 5.
          minimum: 1
          maximum: 20
          default: 5
      
      required:
        - question
    
    endpoint:
      url: http://localhost:8000/rag/query
      method: POST
      headers:
        Content-Type: application/json
    
    response_schema:
      type: object
      properties:
        question:
          type: string
        documents:
          type: array
          items:
            type: object
            properties:
              content:
                type: string
                description: Full parent document content
              metadata:
                type: object
                properties:
                  source:
                    type: string
                  source_file:
                    type: string
        num_results:
          type: integer

  - name: rag_ingest
    description: |
      Add new knowledge to the Mnemonic Cortex incrementally.
      
      Use this tool when you need to:
      - Store meeting summaries or decision records
      - Ingest external research papers or documentation
      - Add newly created protocols or doctrines
      - Capture validated insights or conclusions
      
      Documents are added incrementally without rebuilding the entire database.
      Each document must have a unique doc_id for tracking and deduplication.
    
    parameters:
      type: object
      properties:
        content:
          type: string
          description: |
            Full text content to ingest. Should be well-formatted markdown.
            Include proper headers, sections, and metadata for better retrieval.
          examples:
            - "# Meeting Summary 2024-11-24\n\n## Decisions\n- Approved Protocol 115\n\n## Action Items\n- Implement RAG service"
        
        doc_id:
          type: string
          description: |
            Unique identifier for this document. Use descriptive naming.
            Format: category_topic_date or similar.
          pattern: "^[a-zA-Z0-9_-]+$"
          examples:
            - "meeting_council_20241124"
            - "protocol_115_tactical_mandate"
            - "research_rag_optimization_v2"
      
      required:
        - content
        - doc_id
    
    endpoint:
      url: http://localhost:8000/rag/ingest
      method: POST
      headers:
        Content-Type: application/json
    
    response_schema:
      type: object
      properties:
        status:
          type: string
          enum: [success, error]
        doc_id:
          type: string
        chunks_added:
          type: integer
        error:
          type: string
          nullable: true

# Usage examples for different LLM frameworks
usage_examples:
  ollama:
    language: python
    code: |
      from ollama import Client
      import requests
      
      client = Client()
      tools = [
          {
              'type': 'function',
              'function': {
                  'name': 'rag_query',
                  'description': 'Query the knowledge base',
                  'parameters': {
                      'type': 'object',
                      'properties': {
                          'question': {'type': 'string'},
                          'top_k': {'type': 'integer', 'default': 5}
                      },
                      'required': ['question']
                  }
              }
          }
      ]
      
      response = client.chat(
          model='llama3.1',
          messages=[{'role': 'user', 'content': 'What is Protocol 87?'}],
          tools=tools
      )
  
  langchain:
    language: python
    code: |
      from langchain.tools import StructuredTool
      import requests
      
      def rag_query(question: str, top_k: int = 5) -> dict:
          response = requests.post(
              'http://localhost:8000/rag/query',
              json={'question': question, 'top_k': top_k}
          )
          return response.json()
      
      tool = StructuredTool.from_function(
          func=rag_query,
          name="rag_query",
          description="Query the Mnemonic Cortex knowledge base"
      )
```

### `mcp-rag-service/examples/test_endpoints.sh`

```bash
#!/bin/bash
# test_endpoints.sh - Test script for MCP RAG Service endpoints

set -e

BASE_URL="http://localhost:8000"

echo "=== Testing MCP RAG Service Endpoints ==="
echo ""

# Test 1: Health Check
echo "1. Testing /health endpoint..."
curl -s "$BASE_URL/health" | jq .
echo "✅ Health check passed"
echo ""

# Test 2: Query Endpoint
echo "2. Testing /rag/query endpoint..."
curl -s -X POST "$BASE_URL/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the Anvil Protocol?",
    "top_k": 3
  }' | jq '.num_results'
echo "✅ Query endpoint passed"
echo ""

# Test 3: Ingestion Endpoint
echo "3. Testing /rag/ingest endpoint..."
curl -s -X POST "$BASE_URL/rag/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "# Test Document\n\nThis is a test document for MCP RAG ingestion.",
    "doc_id": "test_doc_'$(date +%s)'"
  }' | jq '.status'
echo "✅ Ingestion endpoint passed"
echo ""

# Test 4: Verify Ingested Content
echo "4. Verifying ingested content can be retrieved..."
curl -s -X POST "$BASE_URL/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "test document MCP RAG ingestion",
    "top_k": 1
  }' | jq '.documents[0].content' | head -n 3
echo "✅ Content retrieval verified"
echo ""

echo "=== All tests passed! ==="
```

---

## 2. Original Deliverables List

1. Complete `mcp-rag-service` project directory with proper structure (`app/`, `Containerfile`, `requirements.txt`)
2. `app/RAG_Engine.py` - Core RAG wrapper class managing ChromaDB, embeddings, and text splitting
3. `app/main.py` - FastAPI application with two endpoints: `POST /rag/query` and `POST /rag/ingest`
4. `Containerfile` - Podman container definition with all dependencies
5. Deployment documentation including Podman build and run commands with persistent volume configuration
6. Integration guide for connecting self-hosted LLM to the MCP RAG Tool Server

## 3. Acceptance Criteria

- FastAPI server successfully builds and runs in a Podman container
- `POST /rag/query` endpoint accepts a question and returns relevant context chunks from ChromaDB
- `POST /rag/ingest` endpoint accepts document content and doc_id, incrementally adds new chunks to existing ChromaDB collection
- ChromaDB data persists across container restarts via host volume mount
- Server is accessible on host network (port 8000) and responds to health checks
- Documentation includes complete setup, deployment, and integration instructions
- Initial test demonstrates successful document ingestion followed by successful query retrieval

## Implementation Details

### Phase 0: Setup, Dependencies, and Container Foundation

**Directory Structure:**
```
mcp-rag-service/
├── Containerfile
├── requirements.txt
├── .env.example
└── app/
    ├── __init__.py
    ├── main.py
    ├── rag_engine.py
    └── config.py
```

**Required Dependencies (`requirements.txt`):**

Based on existing Mnemonic Cortex stack (see `requirements.txt` in project root):

```text
# Python Web Framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0

# RAG Core Components (matching existing Mnemonic Cortex architecture)
langchain>=0.1.0
langchain-community>=0.0.20
langchain-nomic>=0.0.2
chromadb>=0.4.0
nomic[local]  # For local embedding inference

# Utilities
python-dotenv
```

**Environment Configuration (`.env.example`):**
```bash
# ChromaDB Configuration (matching mnemonic_cortex/.env pattern)
CHROMA_ROOT=/app/chroma_data
CHROMA_CHILD_COLLECTION=child_chunks_v5
CHROMA_PARENT_STORE=parent_documents_v5

# Embedding Model
NOMIC_MODEL=nomic-embed-text-v1.5
INFERENCE_MODE=local

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

### Phase 1: Core RAG Logic Wrapper (`app/rag_engine.py`)

**Architecture Alignment:**

This implementation **directly mirrors** the existing Mnemonic Cortex architecture:
- **Reference:** `mnemonic_cortex/app/services/vector_db_service.py` (lines 52-101)
- **Reference:** `mnemonic_cortex/scripts/ingest.py` (lines 88-146)
- **Reference:** `mnemonic_cortex/app/services/embedding_service.py` (lines 22-38)

**Key Components:**

```python
# app/rag_engine.py - Core RAG Engine Implementation
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_classic.storage import LocalFileStore, EncoderBackedStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nomic import NomicEmbeddings
from langchain.schema import Document

class RAGEngine:
    """
    Core RAG engine wrapping existing Mnemonic Cortex architecture.
    
    This class provides the same Parent Document Retriever pattern
    used in the production Mnemonic Cortex, adapted for containerized
    deployment and incremental ingestion.
    
    Architecture:
    - Dual ChromaDB stores: child chunks (searchable) + parent documents (full context)
    - NomicEmbeddings for semantic encoding (local inference mode)
    - ParentDocumentRetriever for context-complete retrieval
    """
    
    def __init__(self, chroma_root: str, child_collection: str, parent_collection: str):
        """
        Initialize RAG engine with persistent ChromaDB stores.
        
        Args:
            chroma_root: Root directory for ChromaDB persistence
            child_collection: Collection name for searchable chunks
            parent_collection: Collection name for full parent documents
        """
        self.chroma_root = Path(chroma_root)
        self.child_collection = child_collection
        self.parent_collection = parent_collection
        
        # Initialize embedding model (singleton pattern from embedding_service.py)
        self.embedding_model = NomicEmbeddings(
            model="nomic-embed-text-v1.5",
            inference_mode="local"
        )
        
        # Initialize text splitter (matching ingest.py line 118)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        
        # Initialize or load existing stores
        self._init_stores()
    
    def _init_stores(self):
        """Initialize ChromaDB vectorstores and parent document store."""
        vectorstore_path = str(self.chroma_root / self.child_collection)
        docstore_path = str(self.chroma_root / self.parent_collection)
        
        # Create directories if they don't exist
        os.makedirs(vectorstore_path, exist_ok=True)
        os.makedirs(docstore_path, exist_ok=True)
        
        # Initialize child chunks vectorstore (matching vector_db_service.py line 75)
        self.vectorstore = Chroma(
            collection_name=self.child_collection,
            persist_directory=vectorstore_path,
            embedding_function=self.embedding_model
        )
        
        # Initialize parent document store (matching vector_db_service.py lines 76-83)
        fs_store = LocalFileStore(root_path=docstore_path)
        self.docstore = EncoderBackedStore(
            store=fs_store,
            key_encoder=lambda k: str(k),
            value_serializer=pickle.dumps,
            value_deserializer=pickle.loads,
        )
        
        # Initialize Parent Document Retriever (matching vector_db_service.py line 87)
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=self.child_splitter
        )
    
    def run_query(self, question: str, top_k: int = 5) -> List[Document]:
        """
        Execute semantic search and return full parent documents.
        
        This method implements the same retrieval pattern as
        mnemonic_cortex/app/services/vector_db_service.py (lines 92-96)
        
        Args:
            question: Natural language query
            top_k: Number of parent documents to return
            
        Returns:
            List of Document objects with full parent content
        """
        results = self.retriever.invoke(question)
        return results[:top_k]
    
    def ingest_document_incremental(self, text: str, doc_id: str) -> Dict[str, Any]:
        """
        Incrementally add new document to existing ChromaDB collection.
        
        CRITICAL: This uses add_documents() NOT rebuild, enabling incremental updates.
        Pattern adapted from ingest.py safe_add_documents() (lines 61-85)
        
        Args:
            text: Full document text content
            doc_id: Unique identifier for source tracking
            
        Returns:
            Dict with status, doc_id, and chunks_added count
        """
        # Create document with metadata
        new_doc = Document(
            page_content=text,
            metadata={"source": doc_id, "source_file": doc_id}
        )
        
        # Add to retriever (handles both chunking and parent storage)
        # This mirrors ingest.py line 139: safe_add_documents(retriever, batch_docs)
        try:
            self.retriever.add_documents([new_doc], ids=None, add_to_docstore=True)
            
            # Calculate chunks for reporting
            chunks = self.child_splitter.split_documents([new_doc])
            chunks_added = len(chunks)
            
            # Persist vectorstore (matching ingest.py line 145)
            self.vectorstore.persist()
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "chunks_added": chunks_added
            }
        except Exception as e:
            return {
                "status": "error",
                "doc_id": doc_id,
                "error": str(e)
            }
```

**Key Design Decisions:**

1. **Parent Document Retriever Pattern:** Maintains existing architecture for context-complete retrieval
2. **Incremental Ingestion:** Uses `add_documents()` instead of rebuilding entire database
3. **Persistent Storage:** ChromaDB data survives container restarts via volume mounts
4. **Singleton Embedding Model:** Efficient resource usage matching existing patterns

### Phase 2: FastAPI API Endpoints (`app/main.py`)

**Architecture Reference:**
- Pattern based on `mnemonic_cortex/app/main.py` query pipeline
- Pydantic validation matching existing service patterns
- Error handling aligned with Protocol 87 standards

**Implementation:**

```python
# app/main.py - FastAPI MCP RAG Tool Server
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from app.rag_engine import RAGEngine

# Load environment configuration
load_dotenv()

app = FastAPI(
    title="MCP RAG Tool Server",
    description="Model Context Protocol RAG service for self-hosted LLMs",
    version="1.0.0"
)

# Initialize RAG Engine (singleton pattern)
rag_engine = RAGEngine(
    chroma_root=os.getenv("CHROMA_ROOT", "/app/chroma_data"),
    child_collection=os.getenv("CHROMA_CHILD_COLLECTION", "child_chunks_v5"),
    parent_collection=os.getenv("CHROMA_PARENT_STORE", "parent_documents_v5")
)

# --- Pydantic Schemas ---

class QueryRequest(BaseModel):
    """Request schema for RAG query endpoint."""
    question: str = Field(..., description="Natural language question to answer")
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)

class DocumentMetadata(BaseModel):
    """Metadata for retrieved document."""
    source: str
    source_file: str

class RetrievedDocument(BaseModel):
    """Retrieved document with content and metadata."""
    content: str
    metadata: DocumentMetadata

class QueryResponse(BaseModel):
    """Response schema for RAG query endpoint."""
    question: str
    documents: List[RetrievedDocument]
    num_results: int

class IngestRequest(BaseModel):
    """Request schema for incremental ingestion endpoint."""
    content: str = Field(..., description="Full document text to ingest")
    doc_id: str = Field(..., description="Unique identifier for document")

class IngestResponse(BaseModel):
    """Response schema for ingestion endpoint."""
    status: str
    doc_id: str
    chunks_added: int = 0
    error: str = None

# --- API Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "service": "mcp-rag-tool"}

@app.post("/rag/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Query the RAG system for relevant context.
    
    This endpoint implements the same retrieval pattern as
    mnemonic_cortex/app/main.py but exposed as an API.
    """
    try:
        # Execute query using RAG engine
        results = rag_engine.run_query(request.question, top_k=request.top_k)
        
        # Format response
        documents = [
            RetrievedDocument(
                content=doc.page_content,
                metadata=DocumentMetadata(
                    source=doc.metadata.get("source", "unknown"),
                    source_file=doc.metadata.get("source_file", "unknown")
                )
            )
            for doc in results
        ]
        
        return QueryResponse(
            question=request.question,
            documents=documents,
            num_results=len(documents)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/rag/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest) -> IngestResponse:
    """
    Incrementally ingest a new document into the RAG system.
    
    This endpoint enables autonomous agent learning by allowing
    LLMs to add knowledge without manual intervention.
    """
    try:
        result = rag_engine.ingest_document_incremental(
            text=request.content,
            doc_id=request.doc_id
        )
        
        return IngestResponse(**result)
    except Exception as e:
        return IngestResponse(
            status="error",
            doc_id=request.doc_id,
            error=str(e)
        )
```

**Key Features:**
- **Health Check:** `/health` endpoint for container monitoring
- **Query Endpoint:** Returns full parent documents with metadata
- **Ingest Endpoint:** Incremental document addition with error handling
- **Pydantic Validation:** Type-safe request/response schemas
- **Error Handling:** Graceful failures with detailed error messages

### Phase 3: Containerization and Deployment

**Containerfile Implementation:**

```dockerfile
# Containerfile - Podman container definition for MCP RAG Service
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY .env.example .env

# Create directory for ChromaDB persistence
RUN mkdir -p /app/chroma_data

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Deployment Commands:**

```bash
# 1. Create persistent storage directory on host
mkdir -p ~/mcp-rag-data

# 2. Build the container image
cd mcp-rag-service
podman build -t mcp-rag-service:latest .

# 3. Run the container with persistent volume
podman run -d \
  --name mcp-rag-tool \
  -p 8000:8000 \
  -v ~/mcp-rag-data:/app/chroma_data:Z \
  -e CHROMA_ROOT=/app/chroma_data \
  mcp-rag-service:latest

# 4. Verify service is running
curl http://localhost:8000/health

# 5. Test query endpoint
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Anvil Protocol?", "top_k": 3}'

# 6. Test ingestion endpoint
curl -X POST http://localhost:8000/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"content": "# Test Document\n\nThis is a test.", "doc_id": "test_doc_001"}'
```

**Container Management:**

```bash
# View logs
podman logs mcp-rag-tool

# Stop container
podman stop mcp-rag-tool

# Start container
podman start mcp-rag-tool

# Remove container (data persists in volume)
podman rm mcp-rag-tool

# Backup ChromaDB data
tar -czf mcp-rag-backup-$(date +%Y%m%d).tar.gz ~/mcp-rag-data
```

### Phase 4: Self-Hosted LLM Integration

**MCP Tool Schema for Function Calling:**

```json
{
  "tools": [
    {
      "name": "rag_query",
      "description": "Query the Mnemonic Cortex knowledge base for relevant context. Use this when you need to answer questions based on Project Sanctuary's canonical documentation, protocols, or chronicles.",
      "parameters": {
        "type": "object",
        "properties": {
          "question": {
            "type": "string",
            "description": "The natural language question to answer"
          },
          "top_k": {
            "type": "integer",
            "description": "Number of documents to retrieve (1-20)",
            "default": 5
          }
        },
        "required": ["question"]
      },
      "endpoint": "http://localhost:8000/rag/query",
      "method": "POST"
    },
    {
      "name": "rag_ingest",
      "description": "Add new knowledge to the Mnemonic Cortex incrementally. Use this when you need to store new information, meeting summaries, or external research for future retrieval.",
      "parameters": {
        "type": "object",
        "properties": {
          "content": {
            "type": "string",
            "description": "The full text content to ingest"
          },
          "doc_id": {
            "type": "string",
            "description": "Unique identifier for this document (e.g., 'meeting_2024_11_24' or 'research_paper_001')"
          }
        },
        "required": ["content", "doc_id"]
      },
      "endpoint": "http://localhost:8000/rag/ingest",
      "method": "POST"
    }
  ]
}
```

**Ollama Integration Example:**

```python
# Example: Integrating MCP RAG with Ollama function calling
import requests
import json
from ollama import Client

client = Client()

# Define tools for Ollama
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'rag_query',
            'description': 'Query the knowledge base for relevant context',
            'parameters': {
                'type': 'object',
                'properties': {
                    'question': {'type': 'string', 'description': 'Question to answer'},
                    'top_k': {'type': 'integer', 'default': 5}
                },
                'required': ['question']
            }
        }
    }
]

# System prompt with RAG awareness
system_prompt = """
You are a sovereign AI assistant with access to the Mnemonic Cortex knowledge base.

When answering questions about Project Sanctuary, protocols, or technical details,
ALWAYS use the rag_query tool to retrieve accurate, canonical information.

Do not rely on your training data for Project Sanctuary specifics - always query the Cortex.
"""

# Example conversation
response = client.chat(
    model='llama3.1',
    messages=[{'role': 'user', 'content': 'What is Protocol 87?'}],
    tools=tools,
    system=system_prompt
)

# Handle tool calls
if response['message'].get('tool_calls'):
    for tool_call in response['message']['tool_calls']:
        if tool_call['function']['name'] == 'rag_query':
            # Call MCP RAG service
            result = requests.post(
                'http://localhost:8000/rag/query',
                json=tool_call['function']['arguments']
            )
            print(f"Retrieved {result.json()['num_results']} documents")
```

## Vision Document Updates

**CRITICAL:** After implementing the MCP RAG Tool Server, update the following vision documents to reflect this new capability:

### 1. Update `mnemonic_cortex/VISION.md`

**Location:** Line 32 (after "Real-Time Mnemonic Writing")

**Add:**
```markdown
*   **✅ MCP RAG Tool Server (Phase 1.5 Complete):** The Cortex has been containerized as an MCP-compliant microservice with incremental ingestion capabilities. AI agents can now autonomously query (`/rag/query`) and ingest (`/rag/ingest`) knowledge through standardized tool calls, enabling true autonomous learning without manual intervention. This completes the Real-Time Mnemonic Writing vision.
```

### 2. Update `mnemonic_cortex/EVOLUTION_PLAN_PHASES.md`

**Location:** Insert new phase between Phase 1 and Phase 2 (after line 27)

**Add:**
```markdown
# -------------------------------------------------------
# ✅ **PHASE 1.5 — MCP RAG Tool Server (COMPLETE)**
# -------------------------------------------------------

**Purpose:**
Transform the Mnemonic Cortex into an autonomous, LLM-accessible microservice with incremental ingestion.

**Deliverables:**
- Containerized FastAPI service with Podman
- `/rag/query` and `/rag/ingest` API endpoints
- Incremental document ingestion without database rebuilds
- Persistent ChromaDB storage via volume mounts
- Self-hosted LLM integration guide

**Status:** ✅ COMPLETE
```

### 3. Update `mnemonic_cortex/RAG_STRATEGIES_AND_DOCTRINE.md`

**Location:** After line 289 (Mnemonic Caching section)

**Add:**
```markdown
### Incremental Ingestion Strategy (MCP RAG Tool Server)

**Problem Solved:** Manual Knowledge Updates and Agent Autonomy

**Mechanism:**
- **MCP API Endpoint:** `POST /rag/ingest` accepts document content and unique doc_id
- **Incremental Processing:** ChromaDB's `add_documents()` appends new vectors without rebuilding
- **Autonomous Learning:** LLMs can expand their knowledge base via tool calls

**Implementation Reference:** `mcp-rag-service/app/rag_engine.py`

**Benefits:**
- Eliminates manual `ingest.py` execution
- Enables autonomous agent learning loops
- Reduces operational overhead
- Supports continuous knowledge expansion
```

### 4. Update `mnemonic_cortex/README.md`

**Location:** After Section 5.2 (line 119)

**Add:**
```markdown
### 5.2.1: MCP RAG Tool Server (Autonomous Ingestion)

For autonomous, incremental knowledge updates via API:

**Start the MCP RAG Service:**
```bash
podman run -d --name mcp-rag-tool -p 8000:8000 \
  -v ~/mcp-rag-data:/app/chroma_data:Z mcp-rag-service:latest
```

**Ingest Documents via API:**
```bash
curl -X POST http://localhost:8000/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"content": "# New Protocol\n...", "doc_id": "protocol_new_v1.md"}'
```

**Query via API:**
```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Anvil Protocol?"}'
```

See **Task #025** for complete implementation and deployment guide.
```

## Notes

### Architecture Alignment
This implementation **directly mirrors** the existing Mnemonic Cortex architecture:
- **Reference:** `mnemonic_cortex/app/services/vector_db_service.py` - Parent Document Retriever pattern
- **Reference:** `mnemonic_cortex/scripts/ingest.py` - Batch ingestion and safe_add_documents logic
- **Reference:** `mnemonic_cortex/core/cache.py` - Caching patterns (future Phase 3 integration)

### Incremental Ingestion Strategy
The key advantage is expanding the knowledge base without expensive rebuilds. ChromaDB's `add_documents()` method appends new vectors to the existing collection, making autonomous knowledge acquisition efficient and scalable.

### MCP Compliance
This implementation uses REST API for immediate usability. Future iterations can adopt full MCP JSON-RPC 2.0 specification for enhanced protocol compliance.

### Security Considerations
For production deployment, consider:
- Adding authentication/authorization to API endpoints
- Implementing rate limiting
- Validating and sanitizing input content
- Using HTTPS for encrypted communication
- Restricting network access via Podman network policies

### Performance Optimization
- Consider batching multiple queries for efficiency
- Implement caching layer for frequently accessed queries (leverage existing `mnemonic_cortex/core/cache.py` patterns)
- Monitor ChromaDB collection size and implement archival strategy if needed
- Profile embedding generation performance and consider GPU acceleration for larger deployments

## Implementation Checklist

Use this checklist to track progress through the four phases:

### Phase 0: Setup ✅
- [ ] Create `mcp-rag-service/` directory structure
- [ ] Copy `requirements.txt` with correct dependencies
- [ ] Create `.env.example` matching project `.env.example` pattern
- [ ] Set up `app/` subdirectory with `__init__.py`, `main.py`, `rag_engine.py`, `config.py`

### Phase 1: RAG Engine ✅
- [ ] Implement `RAGEngine` class in `app/rag_engine.py`
- [ ] Verify Parent Document Retriever pattern matches `vector_db_service.py`
- [ ] Implement `run_query()` method with proper error handling
- [ ] Implement `ingest_document_incremental()` with ChromaDB persistence
- [ ] Test RAG engine independently with sample documents

### Phase 2: FastAPI Endpoints ✅
- [ ] Implement FastAPI app in `app/main.py`
- [ ] Create Pydantic schemas for request/response validation
- [ ] Implement `/health` endpoint for monitoring
- [ ] Implement `POST /rag/query` endpoint
- [ ] Implement `POST /rag/ingest` endpoint
- [ ] Test endpoints locally with `uvicorn`

### Phase 3: Containerization ✅
- [ ] Create `Containerfile` with proper base image and dependencies
- [ ] Build container image with Podman
- [ ] Test container runs and exposes port 8000
- [ ] Configure persistent volume mounting for ChromaDB data
- [ ] Verify data persists across container restarts
- [ ] Test health check endpoint from host

### Phase 4: LLM Integration ✅
- [ ] Create MCP tool schema JSON for function calling
- [ ] Test integration with Ollama (or other self-hosted LLM)
- [ ] Verify LLM can successfully call `/rag/query`
- [ ] Verify LLM can successfully call `/rag/ingest`
- [ ] Test autonomous learning loop (query → ingest → query)
- [ ] Document integration examples for common LLM frameworks

### Documentation Updates ✅
- [ ] Update `mnemonic_cortex/VISION.md` with Phase 1.5 completion
- [ ] Update `mnemonic_cortex/EVOLUTION_PLAN_PHASES.md` with new phase
- [ ] Update `mnemonic_cortex/RAG_STRATEGIES_AND_DOCTRINE.md` with incremental ingestion strategy
- [ ] Update `mnemonic_cortex/README.md` with MCP RAG usage instructions

## Testing Strategy

### Unit Tests
- Test `RAGEngine.run_query()` with various query types
- Test `RAGEngine.ingest_document_incremental()` with different document sizes
- Test Pydantic schema validation with invalid inputs
- Test error handling for missing ChromaDB collections

### Integration Tests
- Test full query pipeline: API request → RAG engine → ChromaDB → response
- Test full ingestion pipeline: API request → RAG engine → ChromaDB persistence
- Test container startup and health check
- Test volume persistence across container restarts

### End-to-End Tests
- Deploy container and test from external LLM client
- Ingest 10+ documents incrementally via API
- Query for ingested content and verify retrieval
- Restart container and verify data persistence
- Test concurrent queries under load

## Success Metrics

The implementation is complete when:

1. **✅ Container Builds:** Podman successfully builds the image without errors
2. **✅ Service Runs:** Container starts and `/health` endpoint returns 200 OK
3. **✅ Query Works:** `/rag/query` returns relevant parent documents with proper metadata
4. **✅ Ingestion Works:** `/rag/ingest` successfully adds documents without rebuilding database
5. **✅ Persistence Works:** ChromaDB data survives container stop/start cycles
6. **✅ LLM Integration Works:** Self-hosted LLM can call both tools successfully
7. **✅ Documentation Updated:** All four vision documents reflect the new capability

## Next Steps After Implementation

Once the MCP RAG Tool Server is operational:

1. **Migrate Existing Data:** Copy existing ChromaDB from `mnemonic_cortex/chroma_db/` to containerized volume
2. **Integrate with Council Orchestrator:** Update council agents to use MCP endpoints
3. **Enable Autonomous Learning:** Configure agents to ingest meeting summaries and decisions
4. **Monitor Performance:** Track query latency, ingestion throughput, and cache hit rates
5. **Plan Phase 2 Integration:** Design how Self-Querying Retriever will integrate with MCP endpoints
6. **Plan Phase 3 Integration:** Design how Mnemonic Caching (CAG) will integrate with containerized service
