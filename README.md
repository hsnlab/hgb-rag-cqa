# HGB-RAG-CQA

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Hierarchical Graph Creation](#hierarchical-graph-creation)
- [Docstring Generation](#docstring-generation)
  - [Baseline](#baseline)
  - [Pure GNN Pipeline](#pure-gnn-pipeline)
  - [Multimodal Pipeline](#multimodal-pipeline)
- [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
  - [Function Clustering (Supernodes)](#function-clustering-supernodes)
  - [Retrieval Process](#retrieval-process)
  - [Answer Generation](#answer-generation)
  - [Repository Support & Data Preparation](#repository-support--data-preparation)
  - [Example: Running the RAG Pipeline](#example-running-the-rag-pipeline)

---

## Introduction

Welcome to the HGB-RAG-CQA project. This repository contains tools and pipelines for hierarchical graph-based code analysis, docstring/comment generation, and retrieval-augmented question answering for code repositories.

---

## Installation

Follow the steps below to set up the environment.  
**Note:** There are separate requirements files for Windows and Linux due to GPU-specific dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/hgb-rag-cqa.git
cd hgb-rag-cqa
```

### 2. Create and activate a virtual environment

**Windows:**
```
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements_windows.txt
```

**Linux:**
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_linux.txt
```

---

## Hierarchical Graph Creation

The core of this project is the hierarchical graph representation of code. This graph consists of:

- **Call Graph (CG) nodes:** Each node represents a function in the codebase.
- **Subgraphs for each function:** For every function node, a subgraph is built representing its internal structure, either as an AST (Abstract Syntax Tree) or a CFG (Control Flow Graph).
- **Edges:** 
  - Between functions (call relationships).
  - Within subgraphs (AST/CFG edges).
  - Hierarchical edges connecting subgraph nodes to their parent function node and vice versa.

**Node Features:**
- Each function node contains:
  - The function's code and metadata.
  - Its docstring and a precomputed embedding (e.g., using Sentence Transformers).
- Each subgraph node contains:
  - The code snippet it represents.
  - An embedding (e.g., using CodeBERT).

**Graph Structure Overview:**
- The main graph is a call graph of the repository.
- Each function node is linked to a subgraph (AST/CFG) representing its internal structure.
- Hierarchical edges connect function nodes and their subgraph nodes, enabling multi-level reasoning.

### Data Preparation

- **Repository files:** Download or clone the target repository you want to analyze into a local folder (e.g., `./repos/Python/manim/manim/`).
- **Notebooks:** For a step-by-step workflow and visualization, see the notebook:
  - [`ericcson-hier-graph-creation-workflow-build.ipynb`](ericcson-hier-graph-creation-workflow-build.ipynb)
- **Python API:** The main logic is implemented in [`package/hierarchical_graph.py`](package/hierarchical_graph.py).

### Example: Building a Hierarchical Graph

```python
from package.hierarchical_graph import HierarchicalGraphBuilder

# Path to the code repository
repo_path = "./repos/Python/manim/manim/"

# Build the hierarchical graph (returns pandas DataFrames)
cg_nodes, cg_edges, sg_nodes, sg_edges, hier_1, hier_2 = HierarchicalGraphBuilder().create_hierarchical_graph(
    path=repo_path,
    return_type="pandas",         # or "pyg" for PyTorch Geometric HeteroData
    repo_functions_only=True,     # only repo functions in CG
    graph_type="CFG",             # or "AST"
    remove_isolated=False,
    remove_subgraph_missing=True,
    batch_size=64
)

# Save to CSV if needed
cg_nodes.to_csv("./graph/manim/cg_nodes.csv", index=False)
cg_edges.to_csv("./graph/manim/cg_edges.csv", index=False)
sg_nodes.to_csv("./graph/manim/sg_nodes.csv", index=False)
sg_edges.to_csv("./graph/manim/sg_edges.csv", index=False)
hier_1.to_csv("./graph/manim/hier_1.csv", index=False)
hier_2.to_csv("./graph/manim/hier_2.csv", index=False)
```

---

## Docstring Generation

Having high-quality docstrings is crucial to maximize our Hierarchical Graoh's potential. Having comprehensive docstrings for all functions and classes is essential for downstream tasks such as code search, or retrieval-augmented generation. 

### Baseline

This script (`baseline.py`) leverages Hugging Face's `datasets` and `transformers` libraries to generate docstrings for every function in a target repository. The script processes each function, removes any existing docstrings, and uses a large language model (LLM) to generate new, well-formatted docstrings following best practices. It then calculates Meteor and BLEU scores when the original docstrings are available, to provide a baseline for the proposed solutions. Finally, it saves the results in a .csv file.

To change the LLM model or the target repository, simply modify the relevant parameters in the `baseline.py` script.

Simply run the script to perform generation with the current settings:

```bash
python baseline.py
```

### Pure GNN Pipeline

The pure GNN pipeline aims to generate docstring embeddings for functions directly from the hierarchical graph structure, without using the original docstrings as input. The process is as follows:

1. **Hierarchical Graph Construction:**  
   - Build the graph detailed in [Data Preparation](#data-preparation).

2. **Docstring Embedding Masking:**  
   - For training, a portion of the function nodes have their docstring embeddings masked (set to zero vectors).
   - The GNN is trained to reconstruct (predict) the masked docstring embeddings using only the graph structure and code embeddings.

3. **GNN Model:**  
   - A multi-layer heterogeneous GNN processes the hierarchical graph, with convolutions on both levels.
   - The model's task is to predict the docstring embedding for each function node, given its local code structure and the surrounding call graph context.

4. **Decoding to Text:**  
   - The predicted docstring embeddings can be decoded back to natural language docstrings using a pretrained decoder (BART).

#### Key Observations & Conclusions

- **High-Dimensional Embeddings:**  
  The docstring embeddings are high-dimensional (300+ or 700+ dimensions), which makes the reconstruction task very challenging for the GNN. The model struggles to accurately reconstruct such large embeddings from graph structure alone.

- **Information Overload:**  
  The nodes contain a lot of information, making it difficult for the GNN to generalize and reconstruct the embeddings effectively.

- **Alternative Approaches:**  
  Simpler node features (e.g., label encodings or just node type definitions) were also tested. These are easier for the GNN to handle but result in less accurate code representations.

- **Conclusion:**  
  The embedding dimension is too large for the GNN to reconstruct accurately, therefore it is not suitable for docstring generation.

For a detailed, runnable example of this pipeline, see [`ericsson-gnn-test.ipynb`](ericsson-gnn-test.ipynb).

### Multimodal Pipeline

The multimodal pipeline for docstring generation combines both code and structural information from the hierarchical graph, inspired by retrieval-augmented generation (RAG) but with a multimodal context. Here, the context for the LLM is constructed from two sources:

- **Code Embedding:** The code is embedded using an attention-based encoder (initially Sentence-BERT, later token-level embeddings from a transformer).
- **Graph Embedding:** Structural information is extracted from the hierarchical graph as the corresponding node embedding (from a GNN).

These two representations are fused—originally by concatenating a 1×384 Sentence-BERT code embedding with the GNN embedding via a linear layer, and later using cross-attention mechanisms. The fused vector is injected as a special token into the LLM's input sequence.

#### Pipeline Evolution

- **Original Setup:**  
  - Code was encoded into a single 384-dimensional vector (Sentence-BERT).
  - Fused with the GNN embedding using a linear layer.
  - Docstring embeddings were averaged for loss calculation.
  - Losses used:
    - Cross-entropy: for token level correctness,
    - Cosine similarity: for semantic corretness.

- **Later Improvements:**  
  - Used attentive pooling for better embedding aggregation (still only one token).
  - Switched to cross-attention fusion (first on Sentence-BERT, then on tokenized code plus GNN embeddings).
  - Explored loss computation both after final response generation and before, with similar results.

#### Key Observations & Conclusions

- Single Token Limitation:  
  Using only one token for the fused context is not sufficient for the LLM to generate high-quality docstrings.
- Fixed Token Count Issues:  
  A fixed, low number of tokens could improve the results, but the compression of original docstrings is an open question.
- Loss Based on Keywords:  
  Loss functions based on keywords are worth exploring as a way of compression for original docstrings.
- GNN-to-LLM Transfer is Hard:  
  It is difficult to teach the LLM to utilize GNN-derived embeddings effectively. This could be due to the limited amount of training data.
- Let LLM Handle Structure:  
  Allowing the LLM to process more of the code structure directly (e.g., by "sequencifying" the graph) may be more effective.
- Graph Sequencing:  
  Traversing the graph to create a sequence (possibly visiting important nodes multiple times, or placing them at the front/back) could improve results (as opposed to simply listing edges and nodes).
    - Could enable the cross-attention mechanism to better attend to structure and code.
    - Necessary for Q/A RAG system as well.
- The hierarchical graph's strength seem limited in docstring generation.

For implementation details and experiments, see the code in [`mvp.py`](mvp.py) and the attention mechanisms in [`attention.py`](attention.py).

To run a model training loop, run:
```bash
python mvp.py
```
The results of the training will be saved into the `/results/` folder. It will contain the loss curve pictures (cosine-, cross entropy-, and combined loss), trained model weigths for the GNN and combination models, and the validation dataframe (in csv format) with generated docstrings and individual BLEU and Meteor scores.

---

## Retrieval-Augmented Generation (RAG)

The RAG pipeline in this project is designed for code question answering by combining semantic search, knowledge-graph-based context filtering, and LLM-based answer generation.

### Function Clustering (Supernodes)

To improve retrieval efficiency and context quality, functions are first clustered based on their docstring or code embeddings. Each cluster acts as a "supernode" representing a group of semantically similar functions on top of the call graph level of the hierarchical graph introduced in the [Data Preparation](#data-preparation) section.

### Retrieval Process

1. **Keyword Extraction:**  
   The user's question is processed to extract key terms using NLP techniques (spaCy library), and given describtions by llm-generation.

2. **Semantic Search:**  
   The extracted keywords are embedded (using SentenceBERT), and cosine similarity is computed against the cluster (supernode) embeddings and class names. The top clusters and classes are selected based on similarity scores.

3. **Candidate Filtering:**  
   Functions belonging to the most relevant clusters and classes are selected as candidates for answering the question.

4. **Context Graph Filtering:**  
   The call graph is filtered to include only the top candidate functions and their structural context. This filtered subgraph is visualized and used as the context for answer generation.

5. **Visualization:**
    As an extra step, the tool creates a visualization of the retrieved subgraph and saves it as `POC2\ filtered_graph.html`.

### Answer Generation

The filtered context (list of relevant functions and their relationships) is provided to a large language model (LLM, e.g., Mistral or similar) as part of the prompt. The LLM then generates a natural language answer to the user's question, grounded in the retrieved code context.

### Repository Support & Data Preparation

- **Supported Repositories:**  
  The current pipeline is set up for the `sklearn` and `manim` repositories only.
- **Large Files:**  
  The `sg_nodes.csv` file (subgraph nodes) is not included in the repository due to its size. After acquiring it, you must manually copy it to `graph/{reponame}` (e.g., `graph/sklearn/sg_nodes.csv`).

### Example: Running the RAG Pipeline

To start the interactive RAG-based code Q&A tool, run:

```bash
python POC2\ kg_rag.py
```
