import os
import json
from typing import Any, List
import bm25s
import Stemmer
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language
)
from src.models.CommandLine import IndexCommand
from src.models.Source import DetailedSource


class VllmIndexing:
    def __init__(self, data: IndexCommand):
        self.file_text: List[Document] = []
        self.vllm_path = data.vllm
        self.config = data

    def create_map(self) -> List[Document]:
        for root, _dirs, files in os.walk(self.vllm_path):
            for file in files:
                if file.endswith('.py') or file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as auto:
                            txt = auto.read()
                            if txt.strip():
                                self.file_text.append(
                                    Document(page_content=txt,
                                             metadata={"path": file_path,
                                                       "filename": file}))
                    except Exception as e:
                        print(f"Erreur lors de la lecture de {file_path}: {e}")
        return self.file_text

    def splitter(self) -> None:
        if not self.file_text:
            self.create_map()

        if not self.file_text:
            raise ValueError("Error: No text files found"
                             f"in path: {self.vllm_path}")

        overlap = 0
        all_chunks_text = []
        chunks_complete_data: List[dict] = []

        for doc in self.file_text:
            fname = doc.metadata["filename"]
            extension = fname.split('.')[-1].lower() if '.' in fname else ""

            lang = self.get_language(extension)
            if lang in [lan.value for lan in Language]:
                chunks = self.codeSplitter(doc.page_content,
                                           lang,
                                           self.config.max_chunk_size,
                                           overlap
                                           )
            else:
                chunks = self.universalSplitter(doc.page_content,
                                                self.config.max_chunk_size,
                                                overlap
                                                )

            for chunk in chunks:
                text = chunk.page_content
                if not text.strip():
                    continue

                first_char_index = chunk.metadata.get("start_index", 0)
                last_char_index = first_char_index + len(text)

                all_chunks_text.append(text)
                chunks_complete_data.append(DetailedSource(
                    chunk_id=len(chunks_complete_data),
                    file_path=doc.metadata["path"],
                    text=text,
                    first_character_index=first_char_index,
                    last_character_index=last_char_index
                    ).model_dump(by_alias=True))

        if not all_chunks_text:
            raise ValueError("Error: Document splitting resulted in "
                             "0 text chunks.")

        stemmer = Stemmer.Stemmer("english")
        corpus_tokens = bm25s.tokenize(all_chunks_text,
                                       stopwords="en",
                                       stemmer=stemmer
                                       )
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        os.makedirs("data/processed", exist_ok=True)
        retriever.save("data/processed/bm25_index")
        output_json_path = "data/processed/chunks_corpus.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(chunks_complete_data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def get_language(extension: str) -> Language | None:
        match extension:
            case "py":
                return Language.PYTHON
            case "md":
                return Language.MARKDOWN
            case "cpp":
                return Language.CPP
            case "java":
                return Language.JAVA
            case "js":
                return Language.JS
            case "ts":
                return Language.TS
            case "php":
                return Language.PHP
            case "html":
                return Language.HTML
            case "cs":
                return Language.CSHARP
            case "c":
                return Language.C
            case _:
                return None

    def universalSplitter(self,
                          text: str,
                          chunk_size: int,
                          chunk_overlap: int
                          ) -> Any:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        return splitter.create_documents([text])

    def codeSplitter(self,
                     text: str,
                     lang: Any,
                     chunk_size: int,
                     chunk_overlap: int
                     ) -> Any:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        return splitter.create_documents([text])
