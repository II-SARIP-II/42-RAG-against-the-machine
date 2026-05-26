from .Source import MinimalSource, DetailedSource
from typing import List
from pydantic import BaseModel


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class DetailedSearchResults(MinimalSearchResults):
    retrieved_sources: List[DetailedSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class DetailedAnswer(DetailedSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentDetailedSearchResults(StudentSearchResults):
    search_results: List[DetailedSearchResults]


class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: List[MinimalAnswer]


class StudentDetailedSearchResultsAndAnswer(StudentSearchResultsAndAnswer):
    search_results: List[DetailedAnswer]
