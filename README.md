# Agentic AI eBook RAG API

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-FF6F00?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Pinecone](https://img.shields.io/badge/Pinecone-v5.0+-000000?logo=pinecone&logoColor=white)](https://www.pinecone.io/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-3.5%20Flash%20Lite-4285F4?logo=google&logoColor=white)](https://aistudio.google.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4.1--Mini-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![Swagger Docs](https://img.shields.io/badge/OpenAPI-3.1.0-85EA2D?logo=swagger&logoColor=black)](http://127.0.0.1:8000/docs)

A production-ready **Corrective & Self-Reflective Agentic RAG (CRAG)** service strictly grounded in the **Agentic AI eBook** (`data/Ebook-Agentic-AI.pdf`). Orchestrated via **LangGraph**, backed by **Pinecone Serverless Vector Database**, and powered by a dual-provider AI engine (**Google Gemini** & **OpenAI**), with an interactive **FastAPI Swagger UI**.

---

## 📖 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Agentic Self-RAG Workflow](#-agentic-self-rag-workflow)
- [Clean Architecture & Modularity](#-clean-architecture--modularity)
- [Multi-Provider Support (Gemini & OpenAI)](#-multi-provider-support-gemini--openai)
- [Strict Document Grounding](#-strict-document-grounding)
- [Directory Structure](#-directory-structure)
- [Prerequisites & Installation](#-prerequisites--installation)
- [Environment Configuration](#-environment-configuration)
- [Document Ingestion Pipeline (ETL)](#-document-ingestion-pipeline-etl)
- [Running the API & Swagger UI](#-running-the-api--swagger-ui)
- [Benchmark Testing & Evaluation](#-benchmark-testing--evaluation)
- [API Specification](#-api-specification)
- [Submission Deliverables Checklist](#-submission-deliverables-checklist)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🚀 Overview & Architecture

Standard RAG implementations use simple linear pipelines (`START ➔ retrieve ➔ generate ➔ END`), blindly passing whatever chunks were retrieved into the LLM. 

This project implements **Corrective Agentic RAG (CRAG)**, where the agent actively evaluates the relevance of retrieved context, short-circuits off-topic questions before wasting generation tokens, verifies hallucination/grounding, and autonomously reformulates queries in a cyclic feedback loop.

```
                     +-------------------+
                     |  LangGraph START  |
                     +---------+---------+
                               |
                               v
                     +-------------------+
                     |   retrieve_node   | <──┐ (Cyclic re-entry from query rewrite)
                     +---------+---------+    │
                               |              │
                               v              │
                     +-------------------+    │
                     |  grade_documents  |    │
                     +---------+---------+    │
                               |              │
                   [ should_generate? ]       │
                     /               \        │
        (Relevant)  /                 \  (Irrelevant / Out-of-scope)
                   v                   v      │
         +-------------------+  +-------------------+
         |   generate_node   |  |   refusal_node    | (Direct exit, 0 generation tokens)
         +---------+---------+  +---------+---------+
                   |                      |
                   v                      v
           [ is_grounded? ]              END
              /          \
         (Valid)      (Low confidence / Refusal)
            /              \
           v                v
          END       +--------------------+
                    | rewrite_query_node |────┘
                    +--------------------+ (max_retries=1)
```

---

## 🧠 Agentic Self-RAG Workflow

The state machine in [`src/graph.py`](file:///c:/My%20Codes/eBook%20RAG/src/graph.py) implements genuine autonomous decision-making:

1. **Vector Retrieval (`retrieve_node`)**:
   - Queries Pinecone for top-$k$ ($k=3$) semantic chunks matching the user's query.
   - Built-in exponential backoff retries handle transient API limits automatically.

2. **Document Relevance Grading (`grade_documents_node`)**:
   - The agent inspects retrieved chunks against the question using structured output (`GradeDocuments`).
   - Assigns a dynamic relevance score and determines whether the content is on-topic.

3. **Conditional Routing (`should_generate`)**:
   - **Branch A (Relevant)**: Routes to `generate_node`.
   - **Branch B (Irrelevant / Out-of-Scope)**: Routes to `refusal_node` immediately.
   - *Token Optimization*: Out-of-scope queries (e.g. *"What is the capital of France?"*) are rejected at the document boundary without calling generation models.

4. **Grounded Generation (`generate_node`)**:
   - Synthesizes an answer strictly from the verified context using a strict system prompt.
   - The LLM self-evaluates its confidence using Pydantic structured output (`GroundedResponse.confidence`).
   - Features automated cascading fallback from primary models to high-efficiency fallback models during traffic overloads.

5. **Hallucination & Grounding Check (`grade_hallucination_and_grounding`)**:
   - If confidence is high ($\ge 0.3$) and no refusal is triggered, completes at `END`.
   - If confidence is low or refusal occurred on the first attempt, triggers autonomous query reformulation.

6. **Cyclic Query Reformulation (`rewrite_query_node`)**:
   - The agent reformulates the user question into an optimized semantic search query tailored for the eBook and **cycles back to `retrieve_node`**.
   - Safeguarded by `retry_count < 1` to strictly eliminate infinite loops.

---

## 🧩 Clean Architecture & Modularity

The codebase adheres to strict separation of concerns:

| Module | Responsibility | Key Components |
| :--- | :--- | :--- |
| **[`src/config.py`](file:///c:/My%20Codes/eBook%20RAG/src/config.py)** | Configuration & Factory | `Settings`, `get_provider_config` factory, Pinecone index discovery cache |
| **[`src/ingestion.py`](file:///c:/My%20Codes/eBook%20RAG/src/ingestion.py)** | Standalone ETL Pipeline | `PyPDFLoader`, `RecursiveCharacterTextSplitter`, dimension safety guard (768 vs 1536) |
| **[`src/graph.py`](file:///c:/My%20Codes/eBook%20RAG/src/graph.py)** | Stateful Corrective LangGraph | `AgentState`, `GradeDocuments`, `GroundedResponse`, conditional edges, cycles |
| **[`api.py`](file:///c:/My%20Codes/eBook%20RAG/api.py)** | FastAPI REST Service | `POST /query`, `GET /health`, root redirect (`/` ➔ `/docs`), OpenAPI schemas |
| **[`app.py`](file:///c:/My%20Codes/eBook%20RAG/app.py)** | Application Entrypoint | Direct runner for the FastAPI service and Swagger server (`python app.py`) |
| **[`tests_sample_queries.py`](file:///c:/My%20Codes/eBook%20RAG/tests_sample_queries.py)** | Benchmark Evaluation | 6 validation queries running in live or offline `--mock` mode |

---

## 🔄 Multi-Provider Support (Gemini & OpenAI)

The application supports dual AI providers configured via the `LLM_PROVIDER` environment variable:

| Feature | Google Gemini (`gemini`) | OpenAI (`openai`) |
| :--- | :--- | :--- |
| **Tier** | Free Tier Testing (Recommended) | Commercial Tier |
| **LLM Model** | `gemini-3.5-flash-lite` (latency ~3s) | `gpt-4.1-mini` |
| **Embedding Model** | `gemini-embedding-2` | `text-embedding-3-small` |
| **Pinecone Dimension** | **`768`** (Matryoshka Truncation) | **`1536`** |
| **Target Index Name** | `agentic-ai-index` | `agentic-ai-index` |
| **Cascading Fallback** | `gemini-2.5-flash` | Exponential Backoff Retry |

> **Matryoshka Representation Note**: Google's `gemini-embedding-2` natively outputs 3072 dimensions. Our pipeline truncates embeddings to 768 dimensions (`output_dimensionality=768`) via Matryoshka Representation Learning, reducing Pinecone serverless vector storage footprint and latency while maintaining semantic search fidelity.

---

## 🛡️ Strict Document Grounding

To eliminate hallucination and prevent out-of-scope answers, the generation prompt enforces:

```text
You are a strict assistant. Answer the question relying ONLY on the context below.
If the context does not contain enough info, state 'I cannot answer based on the provided document.'
Do not use outside knowledge.
```

- **In-Domain Queries**: Answered with high confidence ($\ge 0.85$), citing factual context chunks.
- **Out-of-Domain Queries** (e.g. *"What is the capital of France?"*): Short-circuited at document grading with refusal phrase `"I cannot answer based on the provided document."` and confidence score `0.0`.

---

## 📁 Directory Structure

```text
rag-agentic-ai/
│
├── data/
│   └── Ebook-Agentic-AI.pdf        # Source 60-page knowledge document
│
├── src/
│   ├── __init__.py                 # Package marker
│   ├── config.py                   # Multi-provider factory & environment settings
│   ├── ingestion.py                # Standalone ETL, chunking & Pinecone index setup
│   └── graph.py                    # Corrective LangGraph workflow & state machine
│
├── app.py                          # FastAPI server runner & Swagger entrypoint
├── api.py                          # FastAPI REST API endpoints, schemas & error handlers
├── requirements.txt                # Production Python dependencies
├── .env.example                    # Environment variable template
├── .env                            # Active local credentials (gitignored)
├── README.md                       # Complete technical & setup documentation
└── tests_sample_queries.py         # Automated 6-query benchmark evaluation
```

---

## ⚙️ Prerequisites & Installation

### Requirements
- **Python**: `3.10` or `3.11` (Python 3.11 recommended)
- **RAM**: Minimum 8 GB
- **API Keys**:
  - Pinecone API Key (Free tier from [pinecone.io](https://www.pinecone.io/))
  - Google Gemini API Key (Free tier from [aistudio.google.com](https://aistudio.google.com/)) **OR** OpenAI API Key

### Step 1: Create Virtual Environment

```bash
# Standard venv
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on macOS/Linux:
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Configuration

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Configure your credentials inside `.env`:

```ini
# Active Provider: 'gemini' or 'openai'
LLM_PROVIDER=gemini

# Google Gemini API Key (Free Tier)
GOOGLE_API_KEY=your_google_api_key_here

# OpenAI API Key (if LLM_PROVIDER=openai)
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone Vector DB Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=agentic-ai-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Ingestion Parameters
PDF_PATH=data/Ebook-Agentic-AI.pdf
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=3
```

---

## 📥 Document Ingestion Pipeline (ETL)

Run the offline ETL script to load the PDF, apply semantic chunking, and upsert vectors to Pinecone:

```bash
# Ingest with default provider (Gemini, 768 dimensions):
python -m src.ingestion

# Or ingest specifically for OpenAI (1536 dimensions):
python -m src.ingestion --provider openai

# To drop and recreate an existing index cleanly:
python -m src.ingestion --recreate
```

### Ingestion CLI Options:
| Flag | Description | Default |
| :--- | :--- | :--- |
| `--provider` | Target provider embeddings (`gemini` or `openai`) | Value in `.env` |
| `--recreate` | Drop existing index and recreate fresh with correct dimensions | `False` |
| `--pdf-path` | Path to the PDF source document | `data/Ebook-Agentic-AI.pdf` |
| `--batch-size` | Batch size for vector upserts to Pinecone | `50` |

---

## 🖥️ Running the API & Swagger UI

### 1. Launch the Server

Launch the FastAPI service using either method:

```bash
# Option A: Direct Python runner
python app.py

# Option B: Uvicorn CLI with auto-reload
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Access the Interactive OpenAPI Swagger UI

Once started, open your web browser:

- **Swagger UI (Interactive API Tester)**: [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs)
- **Redoc UI (API Reference Documentation)**: [`http://127.0.0.1:8000/redoc`](http://127.0.0.1:8000/redoc)
- **Direct Root**: Visiting [`http://127.0.0.1:8000/`](http://127.0.0.1:8000/) automatically redirects directly to `/docs`.

### Features of the Swagger Interface:
- **"Try it out" Execution**: Test queries in real-time right inside the browser.
- **Documented Status Codes**: Clear schemas for `200` (Success), `400` (Bad Request), `422` (Validation Error), and `500` (Internal Error).
- **Dedicated Endpoint**: `POST /query` with full request and response schemas.

---

## 🧪 Benchmark Testing & Evaluation

Run the automated benchmark suite against the 6 standardized validation queries defined in the Appening AI assignment brief:

```bash
# Run against live Pinecone index & LLM:
python tests_sample_queries.py

# Run benchmark specifically for OpenAI:
python tests_sample_queries.py --provider openai

# Run in offline mock mode (validates graph, schemas & routing without live API keys):
python tests_sample_queries.py --mock
```

### Official Assignment Validation Queries & Grounding Criteria:

| # | Category | Query | Expected Behavior | Validation Rule |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **Definition & Scope** | *"What is the core definition of Agentic AI as outlined in the eBook?"* | **Grounded** | Synthesizes autonomous, goal-driven systems from text. Score $\ge 0.85$ |
| **2** | **Architecture & Paradigms** | *"What are the main architectural components required to build agentic systems?"* | **Grounded** | Explains Perception, Planning, Memory, Action/Tools. Score $\ge 0.85$ |
| **3** | **Use Cases** | *"What real-world industry use cases for Agentic AI are discussed in the eBook?"* | **Grounded** | Cites industry applications and operations from eBook. Score $\ge 0.85$ |
| **4** | **Comparison** | *"How does Agentic AI differ from traditional generative AI chatbots according to the text?"* | **Grounded** | Differentiates autonomous execution vs passive text generation. Score $\ge 0.85$ |
| **5** | **Challenges & Considerations** | *"What key challenges or limitations of Agentic AI are mentioned in the document?"* | **Grounded** | Discusses governance, complexity, and limitations from text. Score $\ge 0.85$ |
| **6** | **Out-of-Scope Test** | *"What is the capital of France?"* | **Strict Refusal** | Refuses to answer / states info not in document. Score = `0.0` |

### Sample Benchmark Run Output:
```text
================================================================================
AGENTIC AI RAG BENCHMARK EVALUATION
================================================================================
ID  | Query                                  | Type     | Score | Chunks | Result
--------------------------------------------------------------------------------
1   | What is the core definition of Agen... | grounded | 0.98  | 3      | PASS  
2   | What are the main architectural com... | grounded | 0.94  | 3      | PASS  
3   | What real-world industry use cases ... | grounded | 0.94  | 3      | PASS  
4   | How does Agentic AI differ from tra... | grounded | 0.94  | 3      | PASS  
5   | What key challenges or limitations ... | grounded | 0.94  | 3      | PASS  
6   | What is the capital of France?         | refusal  | 0.00  | 0      | PASS  
================================================================================
[SUCCESS] All 6 benchmark test queries passed verification!
```

---

## 📡 API Specification

### Endpoint: `POST /query`

Submits a question to the Agentic RAG pipeline.

#### Request Headers:
```http
Content-Type: application/json
```

#### Request Payload:
```json
{
  "query": "What is the core definition of Agentic AI as outlined in the eBook?",
  "provider": "gemini"
}
```

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `query` | `string` | **Yes** | The user question (min 1 non-whitespace character). |
| `provider` | `string` | No | Optional provider override (`"gemini"` or `"openai"`). Defaults to `LLM_PROVIDER`. |

#### Response Payload (`200 OK`):
```json
{
  "query": "What is the core definition of Agentic AI as outlined in the eBook?",
  "final_answer": "According to the eBook, Agentic AI refers to autonomous systems capable of goal-driven planning, reasoning, memory retention, and tool execution in dynamic environments.",
  "retrieved_context_chunks": [
    "Agentic AI represents autonomous systems combining reasoning, memory, and tool invocation...",
    "While LLMs are powerful tools for processing and generating text, agents are goal-driven systems..."
  ],
  "confidence_score": 0.95
}
```

| Field | Type | Description |
| :--- | :--- | :--- |
| `query` | `string` | The submitted question. |
| `final_answer` | `string` | Synthesized answer strictly grounded in retrieved context. |
| `retrieved_context_chunks` | `array[string]` | Top-$k$ context chunks retrieved from Pinecone. |
| `confidence_score` | `number` | Dynamic score ($0.0$ to $1.0$) reflecting grounding confidence. |

#### Error Responses:
- **`400 Bad Request`**: Malformed request or unsupported provider parameter.
- **`422 Unprocessable Entity`**: Schema validation failure (empty query or whitespace-only).
- **`500 Internal Server Error`**: Sanitized error message preventing internal stack trace leaks.

---

### Endpoint: `GET /health`

Health check endpoint verifying API service and Pinecone configuration.

#### Response (`200 OK`):
```json
{
  "status": "healthy",
  "configured_provider": "gemini",
  "index_name": "agentic-ai-index"
}
```

---

## ✅ Submission Deliverables Checklist

- [x] **Working Python RAG Implementation**: No low-code / vibe-coding tools used.
- [x] **PDF Document Ingestion**: PyPDFLoader and text splitting (`chunk_size=1000`, `chunk_overlap=200`).
- [x] **Pinecone Vector Database**: Serverless index setup with dimension safety checks (768 vs 1536).
- [x] **Stateful LangGraph Workflow**: Corrective RAG with `retrieve`, `grade_documents`, `generate`, and `rewrite_query` nodes.
- [x] **Dynamic Conditional Branching**: `should_generate` and `grade_hallucination_and_grounding` edges with cyclic query rewrite.
- [x] **Strict Document Grounding**: Explicit refusal for out-of-scope questions; non-hardcoded confidence scores.
- [x] **FastAPI & Swagger Documentation**: Production REST service with OpenAPI 3.1.0 schemas and interactive testing at `/docs`.
- [x] **6 Automated Benchmark Queries**: All passing in both live and offline `--mock` modes.
- [x] **Clean Architecture**: Spec Driven Development with zero code coupling.

---

## 🔍 Troubleshooting & FAQs

### 1. `ValueError: Pinecone index dimension mismatch`
- **Cause**: The index was created with Gemini (768 dimensions) and you switched `LLM_PROVIDER=openai` (1536 dimensions), or vice versa.
- **Solution**: Run ingestion with `--recreate` to rebuild the index for the new provider:
  ```bash
  python -m src.ingestion --recreate
  ```

### 2. `ResourceExhausted / 429 Too Many Requests`
- **Cause**: Google Gemini free tier rate limits (15 RPM / 20 RPD on select flash preview models).
- **Solution**: The application uses `gemini-3.5-flash-lite`, which has high free quotas. In addition, built-in cascading logic automatically routes to `gemini-2.5-flash` if transient quota limits are encountered.

### 3. Out-of-Scope Queries Generating Answers
- **Cause**: LLM hallucinating using general knowledge.
- **Solution**: The `grade_documents` node inspects retrieved chunks before generation and short-circuits off-topic questions directly to `refusal_node` with score `0.0`.

### 4. Running Offline Without API Keys
- Run the benchmark in mock mode:
  ```bash
  python tests_sample_queries.py --mock
  ```
  This validates all graph nodes, conditional routing, and output contracts without making external network calls.
