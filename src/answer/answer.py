from src.models.Result import (DetailedAnswer,
                               StudentDetailedSearchResults,
                               StudentDetailedSearchResultsAndAnswer)
from typing import List, Dict, Union, cast
import os.path
import json
import dspy
from .dspy import RAG_sign
from pathlib import Path


class AnswerModel():
    """Configure the LLM backend and DSPy predictor."""
    def __init__(self, model: str):
        """Initialize the LLM client and predictor."""
        self.lm = dspy.LM(
                    model=model,
                    api_base="http://localhost:8000/v1",
                    api_key="EMPTY",
                    max_tokens=256,
                    temperature=0.1,
                    frequency_penalty=0.3,
                    extra_body={"chat_template_kwargs":
                                {"enable_thinking": False}},
                )
        dspy.configure(lm=self.lm)
        self.predictor = dspy.Predict(RAG_sign)


class Answer():
    def __init__(self, question: str, k: int):
        self.question = question
        self.k = k
        self.output_json_path = "data/output/answer_result"
        self.model = AnswerModel("openai/Qwen/Qwen3-0.6B")
        self.past_answer = self.getPastAnswers()

    def findSearchResult(self) -> None:
        path = "data/output/search_results/StudentSearchResults.json"
        if not os.path.isfile(path):
            raise ValueError(f"No search results found at {path}")

        with open(path, 'r', encoding='utf-8') as f:
            searchResultsDict = json.load(f)

        searchResults = StudentDetailedSearchResults.model_validate(
            searchResultsDict
            )
        self.searchResults = searchResults.search_results[0]

    def findChunks(self) -> None:
        """Reads the raw files using character slices from search results."""
        chunks_extracted: List[str] = []

        if not self.searchResults:
            self.context_chunks = []
            return

        target_result = self.searchResults
        for source in target_result.retrieved_sources:
            chunk_text = source.text
            chunks_extracted.append(chunk_text)

        self.context_chunks = chunks_extracted

    def generate_answer(self) -> None:
        res = None
        if os.path.exists(self.output_json_path):
            res = self.getPastAnswers
        if isinstance(res, DetailedAnswer):
            self.minimalAnswer = res
        else:
            full_context = "\n---\n".join(self.context_chunks)
            target_result = self.searchResults

            try:
                result = self.model.predictor(context=full_context,
                                              question=self.question)
                response = result.answer
                response = response.replace("\n", "")
                response = response.replace("[[ ## completed ## ]]", "")

            except dspy.utils.exceptions.ContextWindowExceededError:
                raise ValueError("The k value is too high")
            except Exception:
                raise ValueError("Answering failed")

            minimalAnswer = DetailedAnswer(
                question_id=target_result.question_id,
                question=target_result.question,
                retrieved_sources=target_result.retrieved_sources,
                answer=response
            ).model_dump(by_alias=True)
            self.minimalAnswer = minimalAnswer

    def getPastAnswers(self):
        f1 = Path(self.output_json_path) / "dataset_docs_public.json"
        f2 = Path(self.output_json_path) / "dataset_code_public.json"
        if f1.exists():
            return self.getAnsweredFromFile(f1)
        if f2.exists():
            return self.getAnsweredFromFile(f2)

    def getAnsweredFromFile(self, path: Path) -> Dict[str, int]:
        with open(path, 'r', encoding='utf-8') as f:
            AnswerResultsDict = json.load(f)

        AnswerResults = StudentDetailedSearchResultsAndAnswer.model_validate(
            AnswerResultsDict
            )
        for item in AnswerResults.search_results:
            if item.question == self.question:
                return item
        return None

    def getMinimalAnswer(self) -> None | Dict[str, int]:
        if self.minimalAnswer:
            return self.minimalAnswer
        return None

    def createdAnswerFile(self) -> None:
        os.makedirs(self.output_json_path, exist_ok=True)
        json_path = self.output_json_path + "/answer.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.minimalAnswer, f, indent=4, ensure_ascii=False)
