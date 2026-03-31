# RAG Engine Architecture

---

## 1. High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            INPUT LAYER                                  │
│                                                                         │
│   FinQA PDF Documents                          User Query               │
│   (S&P 500 Annual Reports)                         │                    │
└──────────────┬──────────────────────────────────────┼────────────────────┘
               │                                      │
               ▼                                      ▼
┌─────────────────────────────┐        ┌──────────────────────────────────┐
│      INGESTION PIPELINE     │        │       QUERY REWRITING (optional) │
│                             │        │                                  │
│  PDF Loader                 │        │  • Multi-Query (rule-based)      │
│  → Chunker (3 strategies)   │        │  • Chain-of-Thought (LLM)        │
│  → FAISS  (semantic index)  │        │  • Step-Back (LLM abstraction)   │
│  → Elasticsearch (lexical)  │        │                                  │
└─────────────────────────────┘        └──────────────────┬───────────────┘
               │                                          │
               │                 ┌────────────────────────┘
               ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         HYBRID RETRIEVAL                                │
│                                                                         │
│   FAISS (semantic, k=1) ──┐                                             │
│                            ├──► Fusion (Concat / RRF / CrossEncoder)   │
│   Elasticsearch (BM25 k=3)┘                                             │
└──────────────────────────────────────┬──────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            GENERATION                                   │
│                                                                         │
│   Retrieved Docs → Context Assembly → RAG Prompt → LLM → Answer        │
│                                        (OpenRouter / Gemini)            │
└──────────────────────────────────────┬──────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            EVALUATION                                   │
│                                                                         │
│   Answer ──► Numeric Comparator (tolerance-based)  ──► Correct / Wrong │
│          └─► LLM-as-Judge (AnswerJudge)            ──► Wrong Answer Log │
│                                                         eval_results.csv│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Ingestion Pipeline

```
  PDF Files
      │
      ├──► PyPDFLoader ──────────────────────────┐
      │                                           │
      └──► pdfplumber (table extraction) ─────────┤
                                                  │
                                                  ▼
                                         ┌─────────────────┐
                                         │    Chunking      │
                                         │                  │
                                         │ • Recursive Char │
                                         │ • Fixed Size     │
                                         │ • Sliding Window │
                                         └────────┬─────────┘
                                                  │
                    ┌─────────────────────────────┤
                    │                             │
                    ▼                             ▼
         ┌──────────────────┐         ┌──────────────────────┐
         │ Sentence Transf. │         │   Elasticsearch      │
         │ Embeddings       │         │   (BM25 lexical)     │
         └────────┬─────────┘         └──────────────────────┘
                  │
                  ▼
         ┌──────────────────┐
         │   FAISS Index    │
         │  (per strategy)  │
         └──────────────────┘
```

---

## 3. Retrieval Modes

```
                          Query
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
         faiss_only      es_only        hybrid / rrf / cross_encoder
              │             │                      │
              ▼             ▼              ┌────────┴────────┐
        FAISS only    Elasticsearch       │                 │
        semantic      lexical only     FAISS k=1      ES k=3
          k=1           k=3               │                 │
              │             │             └────────┬────────┘
              │             │                      │
              │             │             ┌────────▼────────────────────┐
              │             │             │  hybrid   → concat + dedup  │
              │             │             │  rrf      → Reciprocal Rank │
              │             │             │             Fusion           │
              │             │             │  cross_enc→ CrossEncoder     │
              │             │             │             Reranker (top-2) │
              │             │             └─────────────────────────────┘
              │             │                      │
              └──────────── ┴──────────────────────┘
                                     │
                                     ▼
                            Retrieved Documents
```

---

## 4. Query Rewriting Strategies

```
  Original Query
        │
        ├─── none ──────────────────────────────────► RAG Chain ──► Answer
        │
        ├─── multi_query ──► Generate N sub-queries ─► RAG Chain ──┐
        │                    (rule-based expansion)   (per query)   │
        │                                                           ├──► Merged
        ├─── chain_of_thought ──► LLM decomposes ───► RAG Chain ──┤    Answer
        │                         into steps          (per step)    │
        │                                                           │
        └─── step_back ──► LLM abstracts ──────────► RAG Chain ──┘
                           to higher-level query     (per step)
```

---

## 5. Evaluation Matrix

```
  EVAL_* env vars
        │
        ▼
  itertools.product over 4 dimensions:
  ┌─────────────────────────────────────────────────────┐
  │  Chunking:   recursive | fixed_size | sliding_window│
  │  Retrieval:  faiss_only | es_only | hybrid | rrf    │
  │  Query:      none | multi_query | cot | step_back   │
  │  Tables:     with_tables | without_tables            │
  └─────────────────────────────────────────────────────┘
        │
        ▼  (for each combination)
  ┌─────────────────────────────────────────────────────┐
  │  Run N FinQA samples (default: 100)                 │
  │                                                     │
  │  ┌─────────────────┐   ┌─────────────────────────┐ │
  │  │ Retrieval       │   │ Answer Accuracy         │ │
  │  │ Precision       │   │                         │ │
  │  │ % docs from     │   │ numeric tolerance check │ │
  │  │ correct source  │   │     OR LLM-as-judge     │ │
  │  └────────┬────────┘   └────────────┬────────────┘ │
  └───────────┼────────────────────────┼───────────────┘
              │                        │
              └───────────┬────────────┘
                          ▼
              logs/eval_results.csv
              logs/retrieval_precision.csv
              logs/wrong_answers/
```

---

## 6. Component Map

```
┌──────────────────────┬──────────────────────────────────────┬───────────────────────────────────────────┐
│ Layer                │ Component                            │ File                                      │
├──────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────┤
│ Entry point          │ main()                               │ main.py                                   │
│ Chat interface       │ chat_interface()                     │ chatbot_interface.py                      │
│ Evaluation harness   │ evaluate_strategies()                │ chatbot_interface.py                      │
│ Ingestion            │ create-vector-store CLI              │ index_store.py                            │
│ Chunking             │ RecursiveCharacter, FixedSize,       │ ingestion/chunking_strategies/            │
│                      │ SlidingWindow                        │                                           │
│ Vector store         │ create_vector_store(),               │ ingestion/vector_store/vector_store.py    │
│                      │ load_vector_store()                  │                                           │
│ FAISS indices        │ recursive, fixed_size, sliding_window│ faiss_indices/                            │
│ Lexical search       │ search_elasticsearch_documents()     │ ingestion/elastic/elasticsearch_utils.py  │
│ Retrieval            │ HybridRetriever,                     │ retrieval_strategies/retriever.py         │
│                      │ CrossEncoderReranker,                │                                           │
│                      │ build_rag_chain()                    │                                           │
│ Query rewriting      │ generate_multi_queries()             │ retrieval_strategies/multi_query.py       │
│                      │ decompose_query_with_cot()           │ retrieval_strategies/chain_of_thought.py  │
│                      │ query_with_step_back()               │ retrieval_strategies/step_back.py         │
│ LLM                  │ build_llm() → OpenRouter / Gemini    │ utils/llm_factory.py                      │
│ Evaluation judge     │ AnswerJudge                          │ evaluator/judges.py                       │
│ Wrong answer log     │ WrongAnswerRecorder                  │ evaluator/wrong_answer_recorder.py        │
│ Cost tracking        │ cost_tracker                         │ utils/cost_tracker.py                     │
│ Rate limiting        │ get_shared_llm_rate_limiter()        │ utils/rate_limiters.py                    │
│ Logging              │ rag_log, log_debug()                 │ logs/                                     │
└──────────────────────┴──────────────────────────────────────┴───────────────────────────────────────────┘
```
