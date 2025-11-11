from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter

def build_rag_chain(retriever):
    rate_limiter = InMemoryRateLimiter(requests_per_second=15/60, max_bucket_size=15)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, rate_limiter=rate_limiter)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
