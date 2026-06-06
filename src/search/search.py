from src.models.Source import DetailedSource
from src.models.Result import (DetailedSearchResults,
                               StudentDetailedSearchResults)
from pathlib import Path
import bm25s
import json
import os
from typing import List, cast, Any, Dict
import chromadb
import dspy
from .queryExpansion import Expansion_sign
import logging


logging.getLogger("dspy").setLevel(logging.ERROR)


class ExpansionModel():
    """Configure the LLM backend and DSPy predictor."""
    def __init__(self, model: str):
        """Initialize the LLM client and predictor."""
        self.lm = dspy.LM(
                    model=model,
                    api_base="http://localhost:8000/v1",
                    api_key="EMPTY",
                    max_tokens=100,
                    temperature=0.0,
                    frequency_penalty=0.3,
                )
        dspy.configure(lm=self.lm)
        self.predictor = dspy.Predict(Expansion_sign)


class Search():
    def __init__(
            self,
            k: int,
            prompt: str,
            save_directory: Path | None,
            chroma: bool,
            expansion: bool,
            questionid: str
            ) -> None:
        self.k = k
        self.prompt = prompt
        self.id = questionid
        self.output_path = save_directory
        self.chroma = chroma
        self.expansion = expansion
        self.long_range_k = 50
        self.model = ExpansionModel("openai/Qwen/Qwen3-0.6B")
        try:
            self.findSources()
        except Exception as e:
            print(e)

    def findSources(self) -> None:
        new_prompt = self.prompt
        if self.expansion:
            try:
                new_prompt += self.query_expansion()
            except Exception:
                pass
        query_tokens = bm25s.tokenize(new_prompt)
        retriever = bm25s.BM25.load("data/processed/bm25_index",
                                    load_corpus=True)
        if self.k > self.long_range_k:
            self.long_range_k = self.k
        docs, _ = retriever.retrieve(query_tokens, k=self.long_range_k)
        bm25_ids = [str(idx) for idx in docs[0]]

        final_ranked_ids = []
        if self.chroma:
            chroma_result = self.semantic_search()
            chroma_ids = (chroma_result.get("ids", [[]])[0]
                          if chroma_result else [])
            final_ranked_ids = self.rank_fusion(bm25_ids, chroma_ids, 60)
            final_ranked_ids = final_ranked_ids[:self.k]
        else:
            final_ranked_ids = bm25_ids[:self.k]

        with open("data/processed/chunks_corpus.json",
                  "r", encoding="utf-8") as f:
            raw_list = json.load(f)
        corpus_dict = {str(item.get("chunk_id")): item for item in raw_list
                       if item.get("chunk_id") is not None}

        sources_formatted: List[DetailedSource] = []
        for chunk_id in final_ranked_ids:
            if chunk_id in corpus_dict:
                validated_source = DetailedSource.model_validate(
                    corpus_dict[chunk_id]
                    )
                sources_formatted.append(validated_source)

        self.sources = sources_formatted

    def query_expansion(self) -> Any:
        try:
            result = self.model.predictor(question=self.prompt)
            response = result.answer.replace("\n", "")
            response = response.replace("[[ ## completed ## ]]", "")
            return response
        except Exception:
            raise ValueError("searching with query expansion failed")

    def semantic_search(self) -> Any:
        client = chromadb.PersistentClient(path="data/processed/chromadb")
        collection = client.get_collection(name="files_content")
        results = collection.query(
            query_texts=[self.prompt],
            n_results=self.long_range_k
        )
        return results

    def rank_fusion(
            self,
            bm25_ids: List[str],
            chroma_ids: List[str],
            k_rrf: int = 60
            ) -> List[str]:
        score_idx: Dict[str, float] = {}

        for i, doc_id in enumerate(bm25_ids):
            rank = i + 1
            score_idx[doc_id] = (score_idx.get(doc_id, 0.0) +
                                 (1.0 / (k_rrf + rank)))

        for i, doc_id in enumerate(chroma_ids):
            rank = i + 1
            score_idx[doc_id] = (score_idx.get(doc_id, 0.0) +
                                 (1.0 / (k_rrf + rank)))

        sorted_ids = [k for k, v in sorted(score_idx.items(),
                                           key=lambda item: item[1],
                                           reverse=True)]
        return sorted_ids

    def findMinimalSearchResults(self) -> None:
        if not self.sources:
            raise Exception("No sources found")
        searchResult = DetailedSearchResults(question_id=self.id,
                                             question=self.prompt,
                                             retrieved_sources=self.sources
                                             ).model_dump(by_alias=True)
        self.detailedSearchResult = searchResult

    def getMinimalSearchResults(self) -> DetailedSearchResults:
        return cast(DetailedSearchResults, self.detailedSearchResult)

    def saveMinimalSearchResults(self) -> None:
        if not self.output_path:
            raise Exception("No ouput path found")
        if not self.detailedSearchResult:
            raise Exception("No MinimalsearchResult found")
        path = str(self.output_path) + "/MinimalSearchResult.json"
        os.makedirs(self.output_path, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.detailedSearchResult,
                      f,
                      indent=4,
                      ensure_ascii=False)

    def findStudentSearchResults(self) -> None:
        self.studentSearchResult = StudentDetailedSearchResults(
            search_results=cast(List[DetailedSearchResults],
                                [self.detailedSearchResult]),
            k=self.k
            ).model_dump(by_alias=True)

    def getStudentSearchResults(self) -> StudentDetailedSearchResults:
        return cast(StudentDetailedSearchResults, self.studentSearchResult)

    def saveStudentSearchResults(self) -> None:
        if not self.studentSearchResult:
            raise Exception("No studentSearchResult found")
        if not self.output_path:
            raise Exception("No output path declared")
        path = str(self.output_path) + "/StudentSearchResults.json"
        os.makedirs(self.output_path, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.studentSearchResult,
                      f,
                      indent=4,
                      ensure_ascii=False)
