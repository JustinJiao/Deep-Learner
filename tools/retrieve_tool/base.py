from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class SearchResult:
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_type: str = ""

class BaseRetriever(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def search(self, query: str, top_k: int) -> List[SearchResult]:
        pass