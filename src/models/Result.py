from .Source import MinimalSource, DetailedSource

from typing import List
from pydantic import BaseModel


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: List[MinimalAnswer]  # type: ignore[assignment]


# Detailed Class
class DetailedSearchResults(MinimalSearchResults):
    retrieved_sources: List[DetailedSource]  # type: ignore[assignment]


class DetailedAnswer(DetailedSearchResults):
    answer: str


class StudentDetailedSearchResults(StudentSearchResults):
    search_results: List[DetailedSearchResults]  # type: ignore[assignment]


class StudentDetailedSearchResultsAndAnswer(StudentSearchResultsAndAnswer):
    search_results: List[DetailedAnswer | None]  # type: ignore[assignment]
