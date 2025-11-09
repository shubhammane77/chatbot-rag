from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

def build_rag_chain(retriever):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
