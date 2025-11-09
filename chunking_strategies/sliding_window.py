from typing import List

from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter

from chunking_strategies.base import ChunkingStrategy


class SlidingWindowChunkingStrategy(ChunkingStrategy):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)
