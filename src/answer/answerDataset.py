from src.models.CommandLine import AnswerDatasetCommand
from src.models.Result import MinimalAnswer, StudentSearchResultsCompleteSource, MinimalSearchResults, MinimalSearchResultsCompleteSource, StudentSearchResultsAndAnswerCompleteSource
from .answer import Answer
import os
import json
from typing import List
from tqdm import tqdm


class AnswerDataset():
    def __init__(self, command: AnswerDatasetCommand):
        self.dataset = command.prompts_file
        self.k = command.k
        self.output_json_path = "data/output/answer_result"

    def findSearchDatasetResult(self) -> None:
        if not os.path.isfile(self.dataset):
            raise ValueError(f"No search results found at {self.dataset}")

        with open(self.dataset, 'r', encoding='utf-8') as f:
            searchDatasetResultsDict = json.load(f)
        searchResult = StudentSearchResultsCompleteSource.model_validate(searchDatasetResultsDict)
        if searchResult.k > self.k:
            self.k = searchResult.k
        answerDataset = []
        for minimalSearchResults in tqdm(searchResult.search_results, desc="LLM answering"):
            answerMinimalSearch = Answer(minimalSearchResults.question, self.k)
            answerMinimalSearch.searchResults = minimalSearchResults
            answerMinimalSearch.findChunks()
            answerMinimalSearch.generate_answer()
            answerDataset.append(answerMinimalSearch.getMinimalAnswer())
        self.searchResultAndAnswer = StudentSearchResultsAndAnswerCompleteSource(
            search_results=answerDataset, 
            k=self.k
        ).model_dump(by_alias=True)

    def createdAnswerDatasetFile(self) -> None:
        if not self.searchResultAndAnswer:
            raise ValueError("No data to put in the output file")
        os.makedirs(self.output_json_path, exist_ok=True)
        json_path = self.output_json_path + "/" + str(self.dataset).split("/")[-1]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.searchResultAndAnswer, f, indent=4, ensure_ascii=False)
