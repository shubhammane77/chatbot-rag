from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
def load_and_split_docs(file_path="/Users/Shubh/Downloads/2507.21004v1.pdf"):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    print(f"Loaded {len(documents)} documents from {file_path}")
    split_docs = splitter.split_documents(documents)
    print(f"Split documents into {split_docs[10].page_content} chunks")
    return split_docs
