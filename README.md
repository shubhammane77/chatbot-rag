# Chatbot RAG

This project implements a Retrieval-Augmented Generation (RAG) chatbot with various chunking and retrieval strategies.

## Setup Instructions

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Shubham-Saini-12/chatbot-rag.git
cd chatbot-rag
```

### 2. Install Dependencies

Use `uv` to install the project dependencies.

```bash
uv sync
```

### 3. Environment Variables

Create a `.env` file in the root directory of the project and add your API keys and document path:

```
GEMINI_API_KEY="your_gemini_api_key_here"
LANGSMITH_API_KEY="your_langsmith_api_key_here"
DOCUMENT_PATH="/path/to/your/document.pdf"
```

### 4. Create Vector Stores

Before running the chatbot, you need to create the FAISS vector stores. This process involves loading the document specified in your `DOCUMENT_PATH` environment variable, applying different chunking strategies, and generating embeddings.

```bash
uv run create-vector-store
```

This command will process the PDF document configured in your `.env` file and create `faiss_index_recursive`, `faiss_index_fixed_size`, and `faiss_index_sliding_window` directories in your project root.

### 5. Inspect Elasticsearch Results (Optional)

After running the vector-store creation command, the same documents are also indexed into Elasticsearch. You can quickly inspect lexical matches with:

```bash
uv run fetch-elasticsearch-results --query "revenue growth" --size 3
```

Override the index with `--index`, and adjust result count with `--size` as needed.

### 6. Run the Chatbot

Once the vector stores are created, you can start the chatbot application:

```bash
uv run chatbot
```

After running the command, the chatbot will prompt you to select a chunking strategy and then you can start asking questions. You will also have options for multi-query generation, chain-of-thought prompting, and step-back decomposition.

### 7. Run Retrieval Precision Evaluation (no LLM required)

Evaluates retrieval precision across all chunking × retrieval mode combinations without calling the LLM. For each query it checks what fraction of retrieved documents come from the correct source file.

```bash
uv run eval-retrieval-precision
```

Control which combinations to test via env vars:

```bash
EVAL_CHUNKING_STRATEGIES=recursive,fixed_size,sliding_window \
EVAL_RETRIEVAL_MODES=faiss_only,es_only,hybrid,rrf,cross_encoder \
EVAL_TABLE_EXTRACTION=with_tables \
EVAL_SAMPLE_COUNT=1000 \
uv run eval-retrieval-precision
```

Results are printed as a table and appended to `logs/retrieval_precision.csv`.

**Retrieval modes:**
| Mode | Description |
|---|---|
| `faiss_only` | FAISS semantic (bi-encoder) search only |
| `es_only` | Elasticsearch BM25 lexical search only |
| `hybrid` | Simple union of both sources (deduped) |
| `rrf` | Reciprocal Rank Fusion — docs in both sources ranked higher |
| `cross_encoder` | Hybrid union re-ranked by a cross-encoder, filtered to top-k |

**Precision metrics reported:**
- **Avg Query Precision** — average of `relevant_docs / total_docs` per query
- **Overall Precision** — `total_relevant / total_retrieved` across all queries

## Project Structure (Overview)

- `main.py`: Main entry point for the chatbot application.
- `create_vector_store_script.py`: Script to create and update FAISS vector stores.
- `vector_store.py`: Contains functions for creating and loading FAISS vector stores.
- `fetch_elasticsearch_results.py`: Utility script to query Elasticsearch directly.
- `retriever.py`: Defines the RAG chain construction.
- `chunking_strategies/`: Contains different document chunking implementations (e.g., `recursive_character.py`, `fixed_size.py`, `sliding_window.py`).
- `retrieval_strategies/`: Contains different query rewriting and decomposition strategies (e.g., `multi_query.py`, `chain_of_thought.py`, `step_back.py`).
- `pyproject.toml`: Project configuration and `uv run` scripts definitions.
- `README.md`: This file, providing setup and project information.
