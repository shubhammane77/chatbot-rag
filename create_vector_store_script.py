import os
from langchain_community.document_loaders import PyPDFLoader
from vectore_store.vector_store import create_vector_store
from chunking_strategies import (
    RecursiveCharacterChunkingStrategy,
    FixedSizeChunkingStrategy,
    SlidingWindowChunkingStrategy,
)
from config import load_environment_variables

def main():
    load_environment_variables()
    print("Creating vector stores...")

    document_path = os.getenv("DOCUMENT_PATH")
    if not document_path:
        print("Error: DOCUMENT_PATH environment variable not set. Please set it in your .env file.")
        exit(1)

    loader = PyPDFLoader(document_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {document_path}")

    chunking_strategies = {
        "recursive": RecursiveCharacterChunkingStrategy(),
        "fixed_size": FixedSizeChunkingStrategy(),
        "sliding_window": SlidingWindowChunkingStrategy(),
    }

    # Create or load vector stores for each strategy
    vector_stores = {}
    for name, strategy in chunking_strategies.items():
        index_name = f"faiss_index_{name}"
        print(f"Creating vector store for {name} strategy...")
        # Here, we pass the raw 'documents' to the chunking strategy, which will split them
        vector_stores[name] = create_vector_store(documents, strategy, index_name=index_name)

if __name__ == "__main__":
    main()
