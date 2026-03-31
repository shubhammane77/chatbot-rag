from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class ChunkingStrategy(ABC):
    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        pass
