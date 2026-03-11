# LangChain + ChromaDB + LangGraph Playground

Hands-on Python examples for building practical GenAI workflows:

- RAG over local documents using ChromaDB
- Conversational agents with session memory
- Multi-step LangGraph pipelines with retry logic
- Local models via Ollama and optional Perplexity integration

## Why this repo

This repository is designed as a learning and reference project for common AI engineering patterns.  
The main knowledge source used across examples is `company_policy.txt`.

## Quick Start (Recommended)

### 1) Clone the project

```bash
git clone https://github.com/<your-username>/langchain_VectorDB_langgraph.git
cd langchain_VectorDB_langgraph
```

### 2) Create and activate virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux / macOS:

```bash
bash setup.sh
```

### 3) Pull Ollama models

```bash
ollama pull llama3.2:3b
ollama pull gemma:2b
```

### 4) Run the first demo

```bash
python Document_chatBot.py
```

## Requirements

- Python 3.10+
- Ollama
- Git

Optional (only for Perplexity demo):

```env
PPLX_API_KEY=your_perplexity_api_key_here
```

## What each script does

| Script | Purpose |
| --- | --- |
| `Document_chatBot.py` | End-to-end RAG over `company_policy.txt` |
| `multi_step_Q&A_workflow.py` | LangGraph pipeline: analyze -> retrieve -> generate |
| `lang_with_history.py` | Conversational chain with per-session memory |
| `langgraph_view.py` | LangGraph typed state and flow demos |
| `get_or_create_collection.py` | Create/upsert/query Chroma collections |
| `task.py` | Similarity search examples with scores |
| `main.py` | Minimal Chroma query example |
| `using_LLM.py` | Ollama translation chain + batch invoke |
| `langchain-setup.py` | Perplexity-powered LangChain translation chain |

## Typical usage

```bash
python Document_chatBot.py
python multi_step_Q&A_workflow.py
python lang_with_history.py
python get_or_create_collection.py
python task.py
python langgraph_view.py
python using_LLM.py
```

Perplexity example (requires `PPLX_API_KEY`):

```bash
python langchain-setup.py
```

## Architecture at a glance

RAG flow:

```text
company_policy.txt
    -> split into chunks
    -> embed with sentence-transformers
    -> store in ChromaDB (./data)
    -> retrieve top-k relevant chunks
    -> generate grounded answer with LLM
```

LangGraph multi-step flow:

```text
User question
    -> analyze (rewrite/search-ready query)
    -> retrieve (Chroma similarity search)
    -> conditional retry when needed
    -> generate final answer
```

## Tech stack

- LangChain, LangGraph
- ChromaDB (persistent local vector store)
- HuggingFace sentence-transformers
- Ollama (`llama3.2:3b`, `gemma:2b`)
- Perplexity AI (`sonar-pro`, optional)
- python-dotenv

## Notes

- This is a CLI/script-based project (no UI yet).
- ChromaDB persistence is stored under `data/`.

## Roadmap

- Add FastAPI or Streamlit interface
- Add persistent chat history store (Redis/SQLite)
- Support PDF/DOCX/web loaders
- Add retrieval reranking and evaluation
- Add Docker setup and graph visualization

## Author

Panendar  
GitHub: https://github.com/Panendar

## License

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
#

