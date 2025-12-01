# PDF FILE PROCESSING - LLM, QA WITH RAG AND AI AGENT

## Architecture and Implementation

---

## SUMMARY

This document describes an AI-based solution for automating the processing, storage, and retrieval of PDF files for technical documents.

The solution leverages open-source technologies including RAG (Retrieval-Augmented Generation), Agent AI, and open-source Large Language Models (LLMs) to enable natural language querying (QA) of technical PDF documents.

---

## 1. OVERVIEW

### 1.1 Core Capabilities

The solution provides two main services:

**Service A: Automated PDF Ingestion and Processing**
- Automatic monitoring of SharePoint folders for new PDF files
- PDF extraction and parsing of technical data (numbers, materials, dimensions, dates)
- Conversion to vector embeddings for semantic search
- Storage in vector database with metadata indexing

**Service B: Intelligent Query and Retrieval Service**
- Natural language interface for certificate queries
- Multi-modal search (text, metadata, semantic similarity)
- AI agent for conversational interaction
- Precise certificate matching and retrieval

### 1.2 Key Technologies

- **LLM**: Llama 3.1 (8B/70B), Mistral 7B, or Qwen 2.5
- **Vector Database**: Qdrant, Weaviate, or Milvus
- **PDF Processing**: Apache Tika, PyMuPDF, Tesseract OCR
- **Embeddings**: sentence-transformers, all-MiniLM-L6-v2
- **Agent Framework**: LangChain or LlamaIndex
- **Document Store**: PostgreSQL with pgvector extension

---

## 2. ARCHITECTURE DESIGN

### 2.1 OPTION A: ON-PREMISES LOCAL SERVER ARCHITECTURE

```
┌────────────────────────────────────────────────────────────────┐
│                      INGESTION PIPELINE                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  SharePoint Folder                                             │
│       │                                                        │
│       ▼                                                        │
│  [Webhook/Polling Service] ──────────────────┐                 │
│       │                                      │                 │
│       ▼                                      │                 │
│  [PDF Processor Service]                     │                 │
│       ├── PDF Parser (PyMuPDF)               │                 │
│       ├── OCR Engine (Tesseract)             │                 │
│       ├── Metadata Extractor (Regex/NER)     │                 │
│       └── Text Chunker                       │                 │
│       │                                      │                 │
│       ▼                                      │                 │
│  [Embedding Service]                         │                 │
│       └── sentence-transformers              │                 │
│       │                                      │                 │
│       ▼                                      │                 │
│  ┌──────────────────────────────┐            │                 │
│  │   Vector Database (Qdrant)   │            │                 │
│  │   - Embeddings               │            │                 │
│  │   - Semantic Search Index    │            │                 │
│  └──────────────────────────────┘            │                 │
│       │                                      │                 │
│       ▼                                      │                 │
│  ┌──────────────────────────────┐            │                 │
│  │   PostgreSQL + pgvector      │            │                 │
│  │   - Original PDFs (blob)     │            │                 │
│  │   - Structured Metadata      │            │                 │
│  │   - Heat Numbers Index       │            │                 │
│  │   - Technical Specs          │            │                 │
│  │   - Timestamps               │            │                 │
│  └──────────────────────────────┘            │                 │
│                                              │                 │
└──────────────────────────────────────────────┘                 │
                                                                 │
┌────────────────────────────────────────────────────────────────┤
│                    QUERY & RETRIEVAL PIPELINE                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  [Web UI / API Gateway]                                        │
│       │                                                        │
│       ▼                                                        │
│  [AI Agent Orchestrator (LangChain)]                           │
│       ├── Query Understanding                                  │
│       ├── Intent Classification                                │
│       ├── Entity Extraction (Heat #, Material, DN, Date)       │
│       └── Query Router                                         │
│       │                                                        │
│       ▼                                                        │
│  [Retrieval Strategy Selector]                                 │
│       ├── Exact Match (Heat Number)                            │
│       ├── Structured Query (SQL)                               │
│       └── Semantic Search (Vector)                             │
│       │                                                        │
│       ▼                                                        │
│  [Hybrid Search Engine]                                        │
│       ├── Vector Search (Qdrant)                               │
│       └── Metadata Filter (PostgreSQL)                         │
│       │                                                        │
│       ▼                                                        │
│  [Re-ranking & Validation]                                     │
│       └── Cross-encoder scoring                                │
│       │                                                        │
│       ▼                                                        │
│  [LLM Response Generator]                                      │
│       ├── Local LLM (Llama 3.1 8B via Ollama)                  │
│       ├── Context: Retrieved documents                         │
│       ├── Response Generation                                  │
│       └── Citation/Source linking                              │
│       │                                                        │
│       ▼                                                        │
│  [Response to User]                                            │
│       └── PDF Download Link                                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 OPTION B: CLOUD-BASED ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                        AZURE CLOUD SERVICES                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SharePoint Online                                              │
│       │                                                         │
│       ▼                                                         │
│  [Azure Logic Apps / Event Grid]                                │
│       │ (Trigger on new file)                                   │
│       ▼                                                         │
│  [Azure Functions - PDF Processor]                              │
│       ├── PDF Extraction (Azure Form Recognizer)                │
│       ├── OCR (Azure Computer Vision)                           │
│       ├── Custom NER (Azure AI Language)                        │
│       └── Text Processing                                       │
│       │                                                         │
│       ▼                                                         │
│  [Azure OpenAI / Azure ML Endpoint]                             │
│       └── Embedding Model (text-embedding-ada-002)              │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────┐                           │
│  │   Azure Cognitive Search         │                           │
│  │   - Vector Search                │                           │
│  │   - Semantic Ranking             │                           │
│  │   - Hybrid Search                │                           │
│  └──────────────────────────────────┘                           │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────┐                           │
│  │   Azure Cosmos DB / Azure SQL    │                           │
│  │   - Document metadata            │                           │
│  │   - Structured indexes           │                           │
│  └──────────────────────────────────┘                           │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────┐                           │
│  │   Azure Blob Storage             │                           │
│  │   - Original PDF files           │                           │
│  └──────────────────────────────────┘                           │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                      QUERY SERVICE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Azure API Management]                                         │
│       │                                                         │
│       ▼                                                         │
│  [Azure Functions - Agent Orchestrator]                         │
│       ├── Semantic Kernel / LangChain                           │
│       ├── Query Processing                                      │
│       └── Intent Recognition                                    │
│       │                                                         │
│       ▼                                                         │
│  [Azure Cognitive Search - Hybrid Retrieval]                    │
│       │                                                         │
│       ▼                                                         │
│  [Azure OpenAI Service]                                         │
│       ├── GPT-4 / GPT-3.5-turbo                                 │
│       ├── RAG Pipeline                                          │
│       └── Response Generation                                   │
│       │                                                         │
│       ▼                                                         │
│  [Web App (React/Vue) - Azure App Service]                      │
│       └── User Interface                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Alternative Cloud Stack (Open Source Focus):
- Kubernetes (AKS) for container orchestration
- Self-hosted Qdrant on Azure VMs
- vLLM serving Llama 3.1 on Azure ML
- PostgreSQL (Azure Database for PostgreSQL)
```

---

## 3. DETAILED COMPONENT SPECIFICATIONS

### 3.1 Service A: PDF Ingestion and Processing Pipeline

#### 3.1.1 SharePoint Monitor
**Purpose**: Detect new PDF files uploaded to SharePoint

**On-Premises Implementation**:
```
Technology: Python + SharePoint REST API
- Libraries: Office365-REST-Python-Client
- Polling interval: 30-60 seconds
- Webhook alternative: Microsoft Graph API webhooks
```

**Cloud Implementation**:
```
Technology: Azure Logic Apps or Event Grid
- Trigger: SharePoint "When a file is created"
- Direct integration with Azure Functions
```

#### 3.1.2 PDF Processing Service - ETL

**Purpose**: Extract text, structure, and metadata (ETL) from PDF files

**Core Components**:

1. **PDF Parser**
   - Library: PyMuPDF (fitz) or Apache Tika
   - Extracts raw text and images
   - Preserves document structure

2. **OCR Engine** (for scanned PDFs)
   - On-Premises: Tesseract OCR 5.x
   - Cloud: Azure Form Recognizer / Computer Vision
   - Handles handwritten or low-quality scans

3. **Metadata Extraction**
   - Number: Regex patterns + NER
   - Code: Pattern matching (e.g., 1.4432)
   - Dimensions: DN (Nominal Diameter) extraction
   - Dates: Purchase date, manufacturing date
   - Supplier information

4. **Text Chunking Strategy**
   - Chunk size: 512 tokens with 50 token overlap
   - Maintains context for embeddings
   - Preserves certificate sections (chemical composition, mechanical properties)


#### 3.1.3 Embedding Service
**Purpose**: Convert text to vector embeddings for semantic search

**On-Premises Models**:
- **all-MiniLM-L6-v2** (384 dimensions, fast, good quality)
- **all-mpnet-base-v2** (768 dimensions, higher quality)
- **multilingual-e5-base** (if multi-language support needed)

**Cloud Models**:
- Azure OpenAI: text-embedding-ada-002 (1536 dimensions)
- Azure OpenAI: text-embedding-3-small (1536 dimensions, newer)

#### 3.1.5 Relational Database (PostgreSQL)
**Purpose**: Store structured metadata and original PDFs

**Schema Design**:

### 3.2 Service B: Query and Retrieval Service

#### 3.2.1 AI Agent Orchestrator
**Purpose**: Manage conversational flow and coordinate retrieval strategies

**Framework Options**:

**Option 1: LangChain (Recommended)**
```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_community.llms import Ollama



**Option 2: LlamaIndex**
```python
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import Ollama

llm = Ollama(model="llama3.1:8b")
service_context = ServiceContext.from_defaults(llm=llm)
index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)

query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("Find certificate for heat number 123456")
```


#### 3.2.4 LLM Response Generation

**On-Premises LLM Options**:

1. **Llama 3.1 8B** (via Ollama or vLLM)
   - Best balance of quality and speed
   - 8GB VRAM minimum
   - Deployment: `ollama run llama3.1:8b`

2. **Mistral 7B v0.3**
   - Excellent instruction following
   - Good for technical queries

3. **Qwen 2.5 7B**
   - Strong reasoning capabilities
   - Multilingual support

**Prompt Template**:
```python
prompt = f"""You are an assistant helping to find PDF files for questions.

User Query: {user_query}

Retrieved Documents:
{retrieved_documents}

Based on the retrieved files, answer the user's question. If you found the exact file(s), 
provide:


Answer:"""
```

**Response Format**:
```json
{
  "answer": "Found the file matching your query...",
  "certificates": [
    {
      "id": "uuid",
      "heat_number": "123456",
      "material_code": "1.4432",
      "dn_size": "DN25",
      "supplier": "Supplier Name",
      "purchase_date": "2020-03-15",
      "confidence_score": 0.95,
      "download_url": "/api/certificates/uuid/download"
    }
  ],
  "sources": ["doc_id_1", "doc_id_2"]
}
```

---

## 4. TECHNOLOGY STACK

### 4.1 On-Premises Stack

**Infrastructure**:
- OS: Ubuntu 22.04 LTS or RHEL 8/9
- Container Runtime: Docker + Docker Compose
- Orchestration (optional): Kubernetes (K3s for small deployments)

**Core Services**:
```yaml
version: '3.8'
services:
  # Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  # Relational Database
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: certificates
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # LLM Inference
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_models:/root/.ollama
    ports:
      - "11434:11434"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # API Service
  api:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      QDRANT_URL: http://qdrant:6333
      POSTGRES_URL: postgresql://admin:${DB_PASSWORD}@postgres:5432/certificates
      OLLAMA_URL: http://ollama:11434
    depends_on:
      - qdrant
      - postgres
      - ollama

  # Ingestion Service
  ingestion:
    build: ./ingestion
    environment:
      SHAREPOINT_CLIENT_ID: ${SP_CLIENT_ID}
      SHAREPOINT_CLIENT_SECRET: ${SP_CLIENT_SECRET}
      SHAREPOINT_SITE_URL: ${SP_SITE_URL}
    depends_on:
      - qdrant
      - postgres

  # Web UI
  web:
    build: ./web
    ports:
      - "3000:3000"
    environment:
      API_URL: http://api:8000

volumes:
  qdrant_data:
  postgres_data:
  ollama_models:
```

**Software Libraries (Python)**:
```
# requirements.txt
langchain==0.1.0
llama-index==0.9.0
sentence-transformers==2.2.2
qdrant-client==1.7.0
psycopg2-binary==2.9.9
pymupdf==1.23.0
pytesseract==0.3.10
transformers==4.35.0
fastapi==0.104.0
uvicorn==0.24.0
pydantic==2.5.0
Office365-REST-Python-Client==2.5.0
python-multipart==0.0.6
httpx==0.25.0
```

**Hardware Requirements (On-Premises)**:
- **Minimum**: 
  - CPU: 8 cores
  - RAM: 32 GB
  - GPU: NVIDIA GTX 1660 Ti (6GB VRAM) or better
  - Storage: 500 GB SSD

- **Recommended**:
  - CPU: 16 cores (AMD EPYC or Intel Xeon)
  - RAM: 64 GB
  - GPU: NVIDIA RTX 4090 (24GB VRAM) or A100
  - Storage: 1 TB NVMe SSD

### 4.2 Cloud Stack (Azure)

**Services**:
- **Compute**: Azure Functions (consumption/premium), Azure App Service
- **Storage**: Azure Blob Storage (PDF files), Azure Cosmos DB or Azure SQL
- **Search**: Azure Cognitive Search (with vector search capabilities)
- **AI**: Azure OpenAI Service or Azure ML (self-hosted models)
- **Integration**: Azure Logic Apps, Event Grid
- **OCR**: Azure Form Recognizer, Azure Computer Vision
- **Monitoring**: Azure Monitor, Application Insights

**Cost Estimation (Monthly)**:
- Small deployment (<1000 certificates): $200-500
- Medium deployment (<10,000 certificates): $500-1500
- Large deployment (>10,000 certificates): $1500-5000

**Alternative Open-Source Cloud Stack**:
- Kubernetes (AKS) + self-hosted components
- Reduces vendor lock-in
- Similar architecture to on-premises

---

## 5. IMPLEMENTATION

### 5.1 Service A Implementation Steps

**Phase 1: Infrastructure Setup (Week 1)**
1. Deploy Docker containers (Qdrant, PostgreSQL, Ollama)
2. Configure SharePoint API access
3. Set up monitoring and logging

**Phase 2: PDF Processing Pipeline (Week 2-3)**
1. Implement SharePoint monitor
2. Build PDF extraction service
3. Develop metadata extraction rules
4. Create embedding service
5. Test with sample certificates

**Phase 3: Database Integration (Week 3-4)**
1. Design database schema
2. Implement data ingestion pipeline
3. Create indexes and optimize queries
4. Build data validation and error handling

**Phase 4: Testing and Optimization (Week 4-5)**
1. Test with real PDF files
2. Fine-tune extraction patterns
3. Optimize embedding performance
4. Load testing

### 5.2 Service B Implementation Steps

**Phase 1: Query Engine Setup (Week 5-6)**
1. Deploy LLM (Ollama with Llama 3.1)
2. Configure LangChain/LlamaIndex
3. Implement basic retrieval functions

**Phase 2: Agent Development (Week 6-7)**
1. Build query understanding module
2. Implement entity extraction
3. Create routing logic
4. Develop hybrid search

**Phase 3: Response Generation (Week 7-8)**
1. Design prompt templates
2. Implement RAG pipeline
3. Build re-ranking mechanism
4. Test conversational capabilities

**Phase 4: User Interface (Week 8-9)**
1. Create web UI (React/Vue)
2. Build API endpoints
3. Implement chat interface
4. Add PDF preview and download

**Phase 5: Integration and Testing (Week 9-10)**
1. End-to-end testing
2. User acceptance testing
3. Performance optimization
4. Documentation

---

## 6. RAG (RETRIEVAL-AUGMENTED GENERATION) PIPELINE

### 6.1 RAG Architecture

```
User Query
    │
    ▼
[Query Preprocessing]
    ├── Tokenization
    ├── Entity Extraction
    └── Query Expansion
    │
    ▼
[Retrieval Phase]
    ├── Vector Search (Semantic)
    ├── Metadata Filter (Structured)
    └── Full-text Search (Keyword)
    │
    ▼
[Candidate Documents]
    │
    ▼
[Re-ranking]
    ├── Cross-encoder scoring
    ├── Relevance filtering
    └── Top-K selection
    │
    ▼
[Context Construction]
    ├── Document assembly
    ├── Deduplication
    └── Context windowing
    │
    ▼
[Generation Phase]
    ├── Prompt engineering
    ├── LLM inference
    └── Response formatting
    │
    ▼
[Post-processing]
    ├── Citation linking
    ├── Confidence scoring
    └── Answer validation
    │
    ▼
Final Response + Sources
```

### 6.2 RAG Implementation

User question: {query}

Provide a clear answer with specific PDF file details. If multiple files match, 
list them all. Include numbers, codes, and other relevant information.

Answer:"""
        
        # Generate response
        response = self.llm.generate(prompt)
        
        return {
            'answer': response,
            'sources': [doc.metadata['document_id'] for doc in retrieved_docs],
            'certificates': [self.format_certificate(doc) for doc in retrieved_docs]
        }
    
    def query(self, user_query):
        # Full RAG pipeline
        retrieved_docs = self.retrieve(user_query)
        response = self.generate(user_query, retrieved_docs)
        return response
```

---

## 7. AGENT AI CAPABILITIES

### 7.1 Agent Architecture

The AI agent provides:
1. **Multi-turn Conversations**: Maintains context across queries
2. **Clarification**: Asks for more details if query is ambiguous
3. **Tool Use**: Decides between exact search, semantic search, or SQL queries
4. **Reasoning**: Explains why certain certificates match

### 7.2 Agent Tools

```python
from langchain.agents import Tool

tools = []

```

### 7.3 Conversational Examples

**Example 1: Exact Number**
```
User: "Find certificate with number 123456"

Agent Reasoning:
1. Detected number: 123456
2. Use Search tool
3. Found 1 exact match

Response: "I found the file for number 123456:
- Supplier: XYZ
- Purchase Date: 2020-03-15
- [Download PDF]"
```
## 8. SECURITY AND COMPLIANCE

### 8.1 Data Security

**On-Premises**:
- TLS/SSL for all communications
- Database encryption at rest (PostgreSQL pgcrypto)
- Access control via RBAC
- VPN access for remote users
- Regular backups (automated daily)

**Cloud**:
- Azure Key Vault for secrets management
- Azure AD for authentication
- Encryption at rest (Azure Storage Service Encryption)
- Encryption in transit (TLS 1.3)
- Network security groups and firewalls

## 9. MONITORING AND MAINTENANCE

### 9.1 System Monitoring

**Metrics to Track**:
- Query response time (target: <2 seconds)
- Ingestion throughput (PDFs/hour)
- Vector search latency
- LLM inference time
- Database query performance
- Storage utilization

**Tools**:
- On-Premises: Prometheus + Grafana, ELK Stack
- Cloud: Azure Monitor, Application Insights

### 9.2 Quality Metrics

**Retrieval Quality**:
- Precision@K (percentage of relevant results in top K)
- Recall (percentage of relevant documents retrieved)
- Mean Reciprocal Rank (MRR)

**User**:
- Success rate (query resolved without clarification)
- User feedback (thumbs up/down)
- Time to find certificate

**Recommendation**: On-premises is more cost-effective for long-term (>3 years) deployments, while cloud offers faster time-to-market and lower upfront costs.

