from src.models.CommandLine import SearchDatasetCommand
from src.models.Question import UnansweredQuestion, AnsweredQuestion
from src.models.Result import StudentDetailedSearchResults
from src.search.search import Search
from pydantic import TypeAdapter
import json
from typing import Union
import os


class SearchDataset():
    def __init__(self, command: SearchDatasetCommand):
        self.k = command.k
        self.dataset_path = command.dataset_path
        self.output_path = command.save_directory
        self.chroma = command.chroma

    def findAllQuestions(self) -> None:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)
        question_adapter: TypeAdapter = TypeAdapter(
            Union[AnsweredQuestion, UnansweredQuestion]
            )
        questions = [question_adapter.validate_python(item)
                     for item in raw_list["rag_questions"]]
        self.questions = questions

    def findQuestionsSources(self) -> None:
        if not self.questions:
            raise Exception("No Questions found")
        searchList = []
        for i, item in enumerate(self.questions):
            search: Search = Search(self.k, item.question, None, self.chroma)
            search.findMinimalSearchResults(question_id=i)
            searchList.append(search.getMinimalSearchResults())
        self.studentSearchResults = StudentDetailedSearchResults(
            search_results=searchList,
            k=self.k
            ).model_dump(by_alias=True)

    def saveSearchDataset(self) -> None:
        if not self.studentSearchResults:
            raise Exception("No minimalSearchsResults found")
        path = (str(self.output_path) + '/' +
                str(self.dataset_path).split('/')[-1])
        os.makedirs(self.output_path, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self.studentSearchResults,
                f,
                indent=4,
                ensure_ascii=False
                )
