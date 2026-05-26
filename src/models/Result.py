from .Source import MinimalSource, DetailedSource
from typing import List
from pydantic import BaseModel
from typing import List
from pydantic import BaseModel
from typing import List, Union
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
    search_results: List[MinimalAnswer]

# Detailed Class

class DetailedSearchResults(MinimalSearchResults):
    retrieved_sources: List[DetailedSource]


class DetailedAnswer(DetailedSearchResults):
    answer: str


class StudentDetailedSearchResults(StudentSearchResults):
    search_results: List[DetailedSearchResults]


class StudentDetailedSearchResultsAndAnswer(StudentSearchResultsAndAnswer):
    search_results: List[Union[DetailedAnswer, None]]
