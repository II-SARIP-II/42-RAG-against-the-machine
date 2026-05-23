from .Source import MinimalSource, CompleteSource
from typing import List
from pydantic import BaseModel


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalSearchResultsCompleteSource(MinimalSearchResults):
    retrieved_sources: List[CompleteSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class MinimalAnswerCompleteSource(MinimalSearchResultsCompleteSource):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsCompleteSource(StudentSearchResults):
    search_results: List[MinimalSearchResultsCompleteSource]


class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: List[MinimalAnswer]


class StudentSearchResultsAndAnswerCompleteSource(StudentSearchResultsCompleteSource):
    search_results: List[MinimalAnswerCompleteSource]
