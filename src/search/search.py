from src.models.CommandLine import SearchCommand
from src.models.Source import CompleteSource
from src.models.Result import MinimalSearchResultsCompleteSource, StudentSearchResultsCompleteSource
from pathlib import Path
import bm25s
import Stemmer
import json
from typing import List
import os


class Search():
    def __init__(self, k: int, prompt: str, save_directory: Path | None, chroma: bool):
        self.k = k
        self.prompt = prompt
        self.output_path = save_directory
        self.chroma = chroma
        try:
            self.findSources()
        except Exception as e:
            print(e)

    def findSources(self):
        query_tokens = bm25s.tokenize(self.prompt)
        retriever = bm25s.BM25.load("data/processed/bm25_index", load_corpus=True)
        docs, scores = retriever.retrieve(query_tokens, k=self.k)
        with open("data/processed/chunks_corpus.json", "r", encoding="utf-8") as f:
            raw_list = json.load(f)
        chunk_ids = list(docs[0])
        sources = []
        for item in raw_list:
            chunk_id = item.get("chunk_id")
            if chunk_id in docs: 
                rank_idx = chunk_ids.index(chunk_id)
                validated_source = CompleteSource.model_validate(item)
                sources.append((rank_idx, validated_source))

        sources.sort(key=lambda x: x[0])
        sources = [source for _, source in sources]
        self.sources = sources

    def findMinimalSearchResults(self, question_id: int = 1):
        if not self.sources:
            raise Exception("No sources found")
        searchResult = MinimalSearchResultsCompleteSource(question_id="q"+str(question_id),
                                            question=self.prompt,
                                            retrieved_sources=self.sources
                                            ).model_dump(by_alias=True)
        self.minimalSearchResult = searchResult

    def getMinimalSearchResults(self):
        return self.minimalSearchResult

    def saveMinimalSearchResults(self):
        if not self.output_path:
            raise Exception("No ouput path found")
        if not self.minimalSearchResult:
            raise Exception("No MinimalsearchResult found")
        path = str(self.output_path) + "/MinimalSearchResult.json"
        os.makedirs(self.output_path, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.minimalSearchResult, f, indent=4, ensure_ascii=False)

    def findStudentSearchResults(self):
        self.studentSearchResult = StudentSearchResultsCompleteSource(
            search_results=[self.minimalSearchResult],
            k=self.k
            ).model_dump(by_alias=True)

    def getStudentSearchResults(self):
        return self.studentSearchResult

    def saveStudentSearchResults(self):
        if not self.studentSearchResult:
            raise Exception("No studentSearchResult found")
        path = str(self.output_path) + "/StudentSearchResults.json"
        os.makedirs(self.output_path, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.studentSearchResult, f, indent=4, ensure_ascii=False)

    def answer(self):
        pass

