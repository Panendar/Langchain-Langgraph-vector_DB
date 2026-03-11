# LangChain · VectorDB · LangGraph

A practical, hands-on collection of AI engineering modules that demonstrates building **RAG (Retrieval-Augmented Generation)** pipelines, **conversational agents with memory**, **multi-step agentic workflows**, and **vector database management** using LangChain, ChromaDB, LangGraph, and locally-hosted LLMs via Ollama.

---

## Project Description

This project serves as an end-to-end reference implementation for core Generative AI engineering patterns:

- Index and search documents using a persistent ChromaDB vector store.
- Build a document Q&A chatbot that answers questions strictly from a provided knowledge base.
- Maintain per-session conversational memory across multiple LLM turns.
- Orchestrate multi-step reasoning workflows using LangGraph state machines with conditional branching and retry logic.
- Integrate both local LLMs (Ollama) and cloud-hosted LLMs (Perplexity AI) through unified LangChain interfaces.

The `company_policy.txt` file (a sample enterprise employee handbook) acts as the primary knowledge document for RAG demonstrations.

---

## Features

- **Document RAG Chatbot** — Chunks, embeds, stores, and retrieves document passages; answers questions using only grounded context.
- **Conversational Memory** — Session-scoped chat history maintained in-memory across multiple interactions.
- **LangGraph Workflows** — Typed state machines with `analyze → retrieve → generate` pipelines, conditional edges, and retry guards.
- **Multi-Step Q&A Workflow** — Query optimization via LLM rewrite, ChromaDB retrieval, document grading, and answer generation in a single graph.
- **Persistent Vector Store** — ChromaDB collections persisted to disk under `./data/` with metadata support.
- **Multi-LLM Support** — Plug-and-play between Ollama models (`llama3.2:3b`, `gemma:2b`) and Perplexity AI (`sonar-pro`).
- **LCEL Pipelines** — LangChain Expression Language chains (`prompt | llm | output_parser`) for translation and general completion tasks.
- **Batch Inference** — Parallel batch invocation of translation chains across multiple language targets.
- **Similarity Search** — Cosine-distance-based document retrieval with scored results.

---

## Tech Stack

| Category              | Technology                          |
| --------------------- | ----------------------------------- |
| **LLM Orchestration** | LangChain, LangGraph                |
| **Local LLM Runtime** | Ollama (`llama3.2:3b`, `gemma:2b`)  |
| **Cloud LLM**         | Perplexity AI (`sonar-pro`)         |
| **Vector Database**   | ChromaDB (persistent)               |
| **Embeddings**        | HuggingFace `sentence-transformers` |
| **Language**          | Python 3.10+                        |
| **Environment**       | python-dotenv                       |

---

## Project Structure

```
langchain_VectorDB_langgraph/
│
├── company_policy.txt            # Sample knowledge-base document (RAG source)
│
├── Document_chatBot.py           # RAG chatbot — loads, chunks, embeds, and answers from company_policy.txt
├── get_or_create_collection.py   # ChromaDB collection creation, upsertion, and semantic querying
├── lang_with_history.py          # Conversational LLM chain with per-session message history
├── langchain-setup.py            # LCEL translation chain using Perplexity AI API
├── langgraph_view.py             # LangGraph basics — state machine, typed state, and RAG state demos
├── main.py                       # Minimal ChromaDB collection query example
├── multi_step_Q&A_workflow.py    # Full LangGraph Q&A pipeline with query rewrite, retrieval & generation
├── task.py                       # ChromaDB similarity search with distance-based scoring
├── using_LLM.py                  # Ollama translation chain with single invoke and batch mode
│
├── data/                         # Persistent ChromaDB storage (auto-generated)
│   ├── chroma.sqlite3
│   └── <collection-uuid>/
│
├── requirements.txt              # Python dependencies
├── setup.sh                      # Shell script to create venv and install dependencies
└── README.md
```

---

## Installation & Setup

### Prerequisites

| Requirement                   | Notes                                           |
| ----------------------------- | ----------------------------------------------- |
| Python 3.10 or newer          | [python.org](https://www.python.org/downloads/) |
| [Ollama](https://ollama.com/) | For running local LLMs                          |
| Git                           | To clone the repository                         |

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/langchain_VectorDB_langgraph.git
cd langchain_VectorDB_langgraph
```

### 2. Create a Virtual Environment

**Linux / macOS:**

```bash
bash setup.sh
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Pull Required Ollama Models

```bash
ollama pull llama3.2:3b
ollama pull gemma:2b
```

### 4. Install Additional LangChain Packages

The `requirements.txt` lists core dependencies. Install the full LangChain ecosystem:

```bash
pip install langchain langchain-community langchain-core langchain-ollama \
            langchain-huggingface langchain-chroma langchain-perplexity \
            langchain-classic langgraph python-dotenv
```

---

## Environment Variables

Create a `.env` file in the project root for optional cloud LLM features:

```env
PPLX_API_KEY=your_perplexity_api_key_here
```

| Variable       | Required                      | Description                                     |
| -------------- | ----------------------------- | ----------------------------------------------- |
| `PPLX_API_KEY` | Only for `langchain-setup.py` | Perplexity AI API key for the `sonar-pro` model |

> All other scripts use **Ollama** (local, no API key required).

---

## Usage

### Document Q&A Chatbot (RAG)

Answers questions strictly from `company_policy.txt`.

```bash
python Document_chatBot.py
```

```
Ask any Question have in your mind (quit/exit to close): What is the standard workday schedule?
Answer: The standard workday begins at 9:30 AM and concludes at 6:30 PM...
```

---

### Multi-Step LangGraph Q&A Workflow

Rewrites the query, retrieves from ChromaDB, and generates a grounded answer.

```bash
python multi_step_Q&A_workflow.py
```

---

### Conversational Agent with Memory

Maintains multi-turn chat history per session ID.

```bash
python lang_with_history.py
```

---

### ChromaDB Collection Management

Create collections, add documents with metadata, and run semantic queries.

```bash
python get_or_create_collection.py
python task.py
python main.py
```

---

### LangGraph State Machine Demo

Explore basic and advanced LangGraph state definitions.

```bash
python langgraph_view.py
```

---

### Translation Chain (Ollama)

Batch-translate text into multiple languages using `llama3.2:3b`.

```bash
python using_LLM.py
```

---

### Translation Chain (Perplexity AI)

Requires `PPLX_API_KEY` in `.env`.

```bash
python langchain-setup.py
```

---

## Architecture Overview

### RAG Pipeline (`Document_chatBot.py`)

```
company_policy.txt
        │
        ▼
 TextLoader → RecursiveCharacterTextSplitter (chunk_size=500, overlap=70)
        │
        ▼
 HuggingFaceEmbeddings → ChromaDB (persisted to ./data)
        │
        ▼
 User Question → RetrievalQA (top-k=3, similarity) → ChatOllama → Answer
```

### LangGraph Multi-Step Workflow (`multi_step_Q&A_workflow.py`)

```
[User Question]
      │
      ▼
  analyze        ← LLM rewrites question into a clean search query
      │
      ▼
  retrieve       ← ChromaDB semantic search (collection: qna_workflow)
      │
      ▼ (conditional: retry if no docs and attempts < 3)
  generate       ← LLM generates answer from retrieved context
      │
      ▼
   [END]
```

---

## Screenshots

> _No UI present — this is a CLI/script-based project._
>
> Add terminal output screenshots here once available.

| Script                       | Expected Output                                                      |
| ---------------------------- | -------------------------------------------------------------------- |
| `Document_chatBot.py`        | ![RAG chatbot screenshot](screenshots/rag_chatbot.png)               |
| `multi_step_Q&A_workflow.py` | ![LangGraph workflow screenshot](screenshots/langgraph_workflow.png) |

---

## Future Improvements

- [ ] **FastAPI / Streamlit UI** — Expose the RAG chatbot as a REST API or interactive web interface.
- [ ] **Persistent Session Memory** — Store conversation history in a database (Redis / SQLite) instead of in-memory dicts.
- [ ] **Advanced Document Loaders** — Support PDF, DOCX, and web URLs via LangChain document loaders.
- [ ] **Reranking** — Add a cross-encoder reranking step after retrieval to improve answer quality.
- [ ] **Streaming Responses** — Implement token-level streaming for real-time chatbot feel.
- [ ] **LLM Evaluation** — Integrate RAGAS or DeepEval for automated RAG pipeline quality metrics.
- [ ] **Docker Support** — Containerize the application with Ollama sidecar for one-command setup.
- [ ] **Multi-Document RAG** — Extend ingestion pipeline to handle entire directories of documents.
- [ ] **Graph Visualization** — Render LangGraph workflow diagrams using Mermaid.

---

## Author

**Venkata** — [GitHub Profile](https://github.com/venkata)

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Venkata

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
#   L a n g c h a i n - L a n g g r a p h - v e c t o r _ D B 
 
 
