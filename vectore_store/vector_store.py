from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

from chunking_strategies.base import ChunkingStrategy

load_dotenv()

def create_vector_store(docs, chunking_strategy: ChunkingStrategy, index_name: str = "faiss_index"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    split_docs = chunking_strategy.split_documents(docs)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(index_name)
    return vector_store

def load_vector_store(index_name: str = "faiss_index"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
