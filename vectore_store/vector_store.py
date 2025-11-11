from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv

from chunking_strategies.base import ChunkingStrategy
import os

load_dotenv()

def create_vector_store(docs, chunking_strategy: ChunkingStrategy, index_name: str = "faiss_index"):
    embeddings = SentenceTransformerEmbeddings(model_name=os.getenv("HF_EMBEDDING_MODEL"))
    split_docs = chunking_strategy.split_documents(docs)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(index_name)
    return vector_store

def load_vector_store(index_name: str = "faiss_index"):
    embeddings = SentenceTransformerEmbeddings(model_name=os.getenv("HF_EMBEDDING_MODEL"))
    return FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
