from src.models.Result import MinimalSearchResultsCompleteSource, MinimalAnswer, StudentSearchResultsCompleteSource 
from typing import List
import logging
import os.path
import json
import ollama


class Answer():
    def __init__(self, question: str, k: int):
        self.question = question
        self.k = k
        self.model_name = "qwen3:0.6b" 
        self.output_json_path = "data/output/answer_result"

    def findSearchResult(self) -> None:
        path = "data/output/search_results/StudentSearchResults.json"
        if not os.path.isfile(path):
            raise ValueError(f"No search results found at {path}")

        with open(path, 'r', encoding='utf-8') as f:
            searchResultsDict = json.load(f)

        searchResults = StudentSearchResultsCompleteSource.model_validate(searchResultsDict)
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
        full_context = "\n---\n".join(self.context_chunks)
        target_result = self.searchResults

        system_instruction = (
            "You are a precise technical assistant answering questions about the vLLM codebase.\n"
            "Your answers must be:\n"
            "1. Self-contained: readable without seeing the original question.\n"
            "2. Source-grounded: rely strictly on the provided context.\n"
            "3. Faithful: do not hallucinate information outside the context.\n"
            "4. Relevant: directly address the query.\n\n"
            "If the context does not contain the answer, state clearly that the information is missing."
        )

        user_prompt = f"Context from codebase:\n{full_context}\n\nQuestion: {target_result.question}"

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": 0.0,
                    "num_predict": 512
                }
            )
            generated_text = response['message']['content'].strip()

        except Exception as e:
            logging.error(f"Failed to generate answer for question {target_result.question_id}: {e}")
            generated_text = "Error: Failed to process generation via local Ollama inference engine."

        minimalAnswer = MinimalAnswer(
            question_id=target_result.question_id,
            question=target_result.question,
            retrieved_sources=target_result.retrieved_sources,
            answer=generated_text
        ).model_dump(by_alias=True)
        self.minimalAnswer = minimalAnswer

    def getMinimalAnswer(self) -> None | MinimalAnswer:
        if self.minimalAnswer:
            return self.minimalAnswer
        return None

    def createdAnswerFile(self) -> None:
        os.makedirs(self.output_json_path, exist_ok=True)
        json_path = self.output_json_path + "/answer.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.minimalAnswer, f, indent=4, ensure_ascii=False)


