# Next Steps

- [-] **Data Preparation:**
    - [-] Configure `DOCUMENT_PATH` in `.env` to point to a directory containing multiple PDF files.
    - [-] Implement a mechanism in `create_vector_store_script.py` to iterate through and process multiple PDF files from the `DOCUMENT_PATH` directory.
- [ ] **Vector Store Creation:**
    - [-] Run `uv run create-vector-store` to create embeddings for all PDF files using various chunking strategies (Recursive Character, Fixed Size, Sliding Window).
- [ ] **Question Answering (QA) Generation:**
    - [ ] Develop a script or modify an existing one to generate a set of benchmark questions and answers based on the content of the embedded PDF files.
    - [ ] Consider using an LLM to assist in generating relevant and diverse questions for each PDF.
- [ ] **Evaluation:**
    - [ ] Implement an evaluation framework to test the RAG chatbot's performance against the generated QA benchmark.
    - [ ] Evaluate the results with different chunking and retrieval strategies (Multi-Query, Chain-of-Thought, Step-Back).
    - [ ] Analyze metrics such as accuracy, relevance, and completeness of answers.
- [ ] **Refinement and Optimization:**
    - [ ] Based on evaluation results, refine chunking parameters or explore new chunking strategies.
    - [ ] Optimize retrieval strategies to improve the quality of retrieved documents.
    - [ ] Experiment with different LLM models or prompting techniques.
