from src.models.CommandLine import (
    IndexCommand, SearchCommand, SearchDatasetCommand,
    AnswerCommand, AnswerDatasetCommand, EvaluateCommand, UserCommand
)
import sys
import fire
from pydantic import ValidationError
from pathlib import Path
from typing import cast


class PipelineCLI:
    """Class that defines CLI commands."""

    def index(
            self,
            max_chunk_size: int = 2000,
            output_directory: Path = Path("data/processed/"),
            vllm: Path = Path("data/raw/vllm-0.10.1")
            ) -> IndexCommand:
        try:
            return IndexCommand(max_chunk_size=max_chunk_size,
                                output_directory=output_directory,
                                vllm=vllm)
        except ValidationError as e:
            print(f"Error: {e}")
            sys.exit(1)

    def search(
            self,
            prompt: str,
            k: int = 5,
            save_directory: Path = Path("data/output/search_results"),
            chroma: bool = False
            ) -> SearchCommand:
        try:
            return SearchCommand(
                prompt=prompt, k=k,
                save_directory=save_directory, chroma=chroma
            )
        except ValidationError as e:
            print(f"Error: {e}")
            sys.exit(1)

    def search_dataset(
            self,
            dataset_path: Path,
            k: int = 5,
            save_directory: Path = Path("data/output/search_results"),
            chroma: bool = False
            ) -> SearchDatasetCommand:
        try:
            return SearchDatasetCommand(
                k=k, dataset_path=dataset_path,
                save_directory=save_directory, chroma=chroma
            )
        except ValidationError as e:
            print(f"Error: {e}")
            sys.exit(1)

    def answer(
            self,
            prompt: str,
            k: int = 1
            ) -> AnswerCommand:
        try:
            return AnswerCommand(
                k=k, prompt=prompt
            )
        except ValidationError as e:
            print(f"Error: {e}")
            sys.exit(1)

    def answer_dataset(
            self,
            prompts_file: Path,
            k: int = 1
            ) -> AnswerDatasetCommand:
        try:
            return AnswerDatasetCommand(
                k=k, prompts_file=prompts_file
            )
        except ValidationError as e:
            print(f"Error: {e}")
            sys.exit(1)

    def evaluate(
            self,
            answer_path: Path,
            dataset_path: Path,
            k: int = 5,
            max_context_length: int = 2000
            ) -> EvaluateCommand:
        try:
            return EvaluateCommand(
                k=k, answer_path=answer_path, dataset_path=dataset_path,
                max_context_length=max_context_length
            )
        except ValidationError as e:
            print(f"Error: {e}")
            sys.exit(1)


def parsing() -> UserCommand:
    return cast(UserCommand, fire.Fire(PipelineCLI))
