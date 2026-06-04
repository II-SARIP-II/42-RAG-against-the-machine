from src.models.Question import UnansweredQuestion, AnsweredQuestion
from src.models.Result import StudentDetailedSearchResults
from src.search.search import Search
from pydantic import TypeAdapter
import json
from typing import Union
import os
from pathlib import Path


class SearchDataset():
    def __init__(
        self,
        dataset_path: Union[str, Path],
        output_dir: Union[str, Path],
        k: int,
        chroma: bool,
        expansion: bool
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.k = k
        self.chroma = chroma
        self.expansion = expansion

    def findAllQuestions(self) -> None:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)
        q_adapter: TypeAdapter[
            Union[AnsweredQuestion, UnansweredQuestion]] = TypeAdapter(
            Union[AnsweredQuestion, UnansweredQuestion]
        )
        questions = [q_adapter.validate_python(item)
                     for item in raw_list["rag_questions"]]
        self.questions = questions

    def findQuestionsSources(self) -> None:
        if not self.questions:
            raise Exception("No Questions found")
        searchList = []
        for i, item in enumerate(self.questions):
            search: Search = Search(k=self.k,
                                    prompt=item.question,
                                    save_directory=None,
                                    chroma=self.chroma,
                                    questionid=item.question_id,
                                    expansion=self.expansion)
            search.findMinimalSearchResults()
            searchList.append(search.getMinimalSearchResults())
        self.studentSearchResults = StudentDetailedSearchResults(
            search_results=searchList,
            k=self.k
            ).model_dump(by_alias=True)

    def saveSearchDataset(self) -> None:
        if not self.studentSearchResults:
            raise Exception("No minimalSearchsResults found")
        path = (str(self.output_dir) + '/' +
                str(self.dataset_path).split('/')[-1])
        os.makedirs(self.output_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self.studentSearchResults,
                f,
                indent=4,
                ensure_ascii=False
                )
