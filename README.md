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

**HGB-RAG-CQA** is a toolkit for hierarchical graph-based code analysis and retrieval-augmented question answering (RAG) on code repositories. It supports semantic search, knowledge graph retrieval, and LLM-based answer generation.

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

## Repository Structure

- `run.py`  
  Main entry point for interactive RAG-based code Q&A.

- `src/`  
  Source code folder containing all modules:
  - `rag/`  
    Core RAG logic (`base_rag.py`, `simple_rag.py`, `repo_rag.py`, `config.py`)
  - `eval/`  
    Evaluation scripts and QA generation (`evaluation.py`, `metrics.py`, etc.)
  - `utils/`  
    Utility modules for retrieval, reranking, deduplication, and vector store management.

- `requirements_windows.txt`, `requirements_linux.txt`  
  OS-specific dependencies.

---

## Running the Code

### Interactive RAG Search

To start the interactive code Q&A tool, run:

```bash
python run.py
```

You will be prompted for a question. The system will retrieve relevant code/documentation, rerank results, and generate an answer using an LLM.

### Evaluation

To evaluate the framework, you can run an ablation study using `src/run_ablation.py`. The script sweeps the parameter grid defined within the code and saves the results using mlflow. Example usage:
```bash
python src/run_ablation.py --eval_data_path data/yourdata.csv --qdrant_collection your_qdrant_collection_name --qdrant_api_key your_qdrant_api_key --quantize_llm
```

---

## Notes

- The repository supports both Windows and Linux.
- For advanced usage, see the source code in `src/` for customization via rag config.