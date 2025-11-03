# HGB-RAG-CQA

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Running the Code](#running-the-code)
  - [Interactive RAG Search](#interactive-rag-search)
  - [Evaluation and QA Generation](#evaluation-and-qa-generation)
- [Notes](#notes)

---

## Introduction

**HGB-RAG-CQA** is a modular toolkit for retrieval-augmented question answering (RAG) over code repositories that integrates knowledge-graph context with semantic vector search and LLM generation. The project combines a Neo4j-based code knowledge graph, a Qdrant-backed vector store, and configurable LLM pipelines to retrieve, rerank, and synthesize answers to developer-oriented queries (e.g., about functions, issues, or PRs).

This repository builds upon the code knowledge graph construction produced by the hsnlab/code-knowledge-graph project (https://github.com/hsnlab/code-knowledge-graph), and uses that KG as the graph backbone for graph-aware retrieval and expansion.

Key features:
- Hybrid retrieval: graph-aware expansion + semantic search for robust candidate selection.
- Metadata-aware reranking and deduplication to surface high-quality, relevant documents.
- Agentic and pipeline-based LLM integration for both simple Q&A and multi-step reasoning flows.
- Evaluation suite with retrieval and generation metrics to run experiments and ablations.
- Config-driven setup and cross-platform support (Windows / Linux).

Usage targets: researchers and engineers who want reproducible RAG experiments on codebases and KG-enhanced retrieval for developer assistance.

---

## Installation

Clone the repository and set up a Python virtual environment.  
**Note:** Use the requirements file matching your operating system.

### 1. Clone the Repository

```bash
git clone https://github.com/hsnlab/hgb-rag-cqa.git
cd hgb-rag-cqa
```

### 2. Create and Activate a Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements_windows.txt
```

**Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_linux.txt
```

---

## Configuration (must-have)

This project expects a small configuration directory named `_` at the repository root. Place API tokens and JSON config files there. The repo includes example filenames; fill them with your credentials / settings and do NOT commit secrets.

Files you will find / should create in `_/`:
- hf_token.txt - Hugging Face token (plain text).
- github_token.txt - GitHub API token (plain text).
- qdrant_api_key.txt - Qdrant API key (plain text) - name may vary; can also be set in qdrant_config.json.
- qdrant_config.json - Qdrant connection & collection settings.
- neo4j_config.json - Neo4j connection settings (uri, user, password, database).
- rag_config.json - RAG pipeline parameters and toggles (Passed to `PipelineConfig` in `src/rag/config.py`).

Place the files under the repository `_` directory.

Minimal example - qdrant_config.json
```json
{
  "url": "https://localhost:6333",
  "api_key": "password",      // optional if using qdrant_api_key.txt
  "collection": "functions_v1",
  "embedding_model": "all-MiniLM-L6-v2",
  "distance": "Cosine",
  "port": 6333
}
```

Minimal example - neo4j_config.json
```json
{
  "uri": "bolt://localhost:7687",
  "user": "neo4j",
  "password": "your_neo4j_password",
  "database": "neo4j"
}
```

Notes and recommendations:
- For security, omit `_` directory from version control.
- qdrant_config.json may include the api_key for convenience; the code typically supports reading the key from the file in `_` if you prefer separation.
- Ensure the Qdrant collection's vector dimension matches your embedding model; mismatches cause errors.
- Update rag_config.json to tune retrieval/reranking/deduplication/generation behavior for experiments or evaluation.

---

## Repository Structure

- `run.py`  
  Main entry point for interactive non-agentic RAG-based code Q&A.

- `run_agentic.py`  
  Launcher for the agentic RAG flows that orchestrate multi-step reasoning with LangGraph workflows. There are too options available, a "free" single agent setup, and a "strict" one, with well defined steps (retrieval, graph-based context building, etc.,)

- `src/`  
  Source modules:

  - `rag/`  
    Core RAG implementations and pipeline configs:
    - Non-agentic pipelines:
      - `repo_rag.py`, `simple_rag.py`, `base_rag.py` -  RAG flows that run retrieval, reranking, deduplication, and generate an answer in one pass.
      - Use these for straightforward QA and evaluation where a single LLM prompt suffices.
    - Agentic pipelines:
      - `agentic_langgraph*.py` - multi-step agentic flows that orchestrate multiple tools/agents (e.g., graph walkers, code reasoners, iterative retrieval). These expose a agent interface (used by `run_agentic.py`) designed for complex, multi-turn or stepwise reasoning.

  - `eval/`  
    Evaluation and QA generation scripts:
    - `kg_rag_eval.py`, `kg_rag_eval_collections.py`, `evaluation.py`, `metrics.py`, `generate_qa.py`, etc.
    - Use these for batch evaluation, ablations, and metric computation.

  - `utils/`  
    Helpers and building blocks:
    - `qdrant_store.py` - vector store adapter.
    - `retrieval.py`- retrieval connector.
    - `reranker.py`, `deduplicator.py`, `graph_context.py` - post-retrieval processing and context assembly.
    - `config_loader.py`, `langgraph_utils.py` - config and LangGraph helpers.
    - `neo4j_mcp_server.py`, `qdrant_mcp_server.py` MCP servers to expose tools using Qdrant and Neo4J servers to agents.

- `_/`  
  Configuration directory (tokens and JSON configs). See Configuration section for expected files.

- `requirements_windows.txt`, `requirements_linux.txt`  
  OS-specific dependency lists.

Notes:
- Choose non-agentic pipelines for lower-latency, single-step answers and easy integration with evaluation scripts.
- Choose agentic pipelines when you need multi-step tool use, iterative KG exploration, or complex reasoning that benefits from an agent.

---

## Running the Code

This repo provides non-agentic (single-shot) and agentic (multi-step) RAG flows, evaluation scripts, and utilities to populate the vectorstore from a Neo4j code knowledge graph.

### Prerequisites
- Fill `_/*` config files (see Configuration section) before running.
- Qdrant server and Neo4j server must be accessible and configured in `_/qdrant_config.json` and `_/neo4j_config.json`.
- Hugging Face / model tokens in `_ /hf_token.txt` when needed.

Populating the vectorstore from Neo4j
- The repo includes notebooks and helpers to index KG content into Qdrant (see `create-vectorstore.ipynb` and `src/utils/qdrant_store.py`).
- The KG should be the graph built with the hsnlab/code-knowledge-graph project (https://github.com/hsnlab/code-knowledge-graph). Build / export that KG into your Neo4j instance first, then use this repo to embed and index nodes.
- Quick steps:
  1. Ensure Neo4j contains the code KG (hsnlab/code-knowledge-graph).
  2. Configure `_/qdrant_config.json` and `_/neo4j_config.json`.
  3. Run the `create-vectorstore.ipynb` notebook.
  4. Verify with `store.get_collection_info()`.

### Non-agentic (interactive) pipeline
- Use the main non-agentic CLI:
  ```bash
  python run.py
  ```
- Feel free to experiment with configuration settings in `_/rag_config.json`.

### Agentic (multi-step) pipeline
- Use the main Agentic (LangGraph) CLI:
  ```bash
  python -m run_agentic --pipeline <pipeline option> --model-name <your ollama model id>
  ```
  - test_langgraph.py — interactive tester for agentic flows (recommended for development).
  - `--pipeline` can be `free` (less constrained) or `strict` (predefined steps).

### Batch evaluation and experiments
- Evaluation scripts live in `src/eval/`. Use them for batch metrics, QA generation, and ablation studies.
  - Run a retrieval+generation evaluation:
    ```bash
    python -m src.eval.kg_rag_eval --eval-path "data/your_eval_set.csv"
    ```
    - Use `python -m src.eval.kg_rag_eval --help` to see all of the available arguments the script takes

---

## Notes

- The repository supports both Windows and Linux.

- For advanced usage, see the source code in `src/` for customization via rag config.

- Some of the scripts use api keys to Huggingface and GitHub. To ensure the code runs smoothly, create two separate files for these API keys at `/_/hf_token.txt` and `/_/github_token.txt` containing the respective access tokens.
