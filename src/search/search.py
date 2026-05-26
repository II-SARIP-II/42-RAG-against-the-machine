from src.models.Source import DetailedSource
from src.models.Result import (DetailedSearchResults,
                               StudentDetailedSearchResults)
from pathlib import Path
import bm25s
import json
import os
from typing import List, cast
import chromadb


class Search():
    def __init__(
            self,
            k: int,
            prompt: str,
            save_directory: Path | None,
            chroma: bool
            ) -> None:
        self.k = k
        self.prompt = prompt
        self.output_path = save_directory
        self.chroma = chroma
        try:
            self.findSources()
        except Exception as e:
            print(e)

    def findSources(self) -> None:
        query_tokens = bm25s.tokenize(self.prompt)
        if self.chroma:
            chroma_result = self.semantic_search()
            print(chroma_result)
        retriever = bm25s.BM25.load("data/processed/bm25_index",
                                    load_corpus=True)
        docs, _ = retriever.retrieve(query_tokens, k=self.k)
        print(docs, _)
        with open("data/processed/chunks_corpus.json",
                  "r", encoding="utf-8") as f:
            raw_list = json.load(f)
        chunk_ids = list(docs[0])
        sources = []
        for item in raw_list:
            chunk_id = item.get("chunk_id")
            if chunk_id in docs:
                rank_idx = chunk_ids.index(chunk_id)
                validated_source = DetailedSource.model_validate(item)
                sources.append((rank_idx, validated_source))

        sources.sort(key=lambda x: x[0])
        sources_fomatted: List[DetailedSource] = [s for _, s in sources]
        self.sources = sources_fomatted

    def semantic_search(self):
        client = chromadb.PersistentClient(path="data/processed/chromadb")
        collection = client.get_collection(name="files_content")
        results = collection.query(
            query_texts=[self.prompt],
            n_results=self.k
        )

        return results

    def findMinimalSearchResults(self, question_id: int = 0) -> None:
        if not self.sources:
            raise Exception("No sources found")
        searchResult = DetailedSearchResults(question_id="q"+str(question_id),
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
