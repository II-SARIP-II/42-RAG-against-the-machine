from pydantic import BaseModel, Field, FilePath, ConfigDict
from pathlib import Path
from enum import Enum
from typing import Literal, Union, Annotated


class Actions(str, Enum):
    INDEX = "index"
    SEARCH = "search"
    SEARCH_DATASET = "search_dataset"
    ANSWER = "answer"
    ANSWER_DATASET = "answer_dataset"
    EVALUATE = "evaluate"


class UserCommand(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    action: Actions


class IndexCommand(UserCommand):
    action: Literal[Actions.INDEX] = Actions.INDEX
    max_chunk_size: int = Field(default=2000)
    output_directory: Path = Field(default=Path("data/processed/"))
    vllm: Path = Field(default=Path("data/raw/vllm-0.10.1"))
    chroma: bool = Field(default=False)


class SearchCommand(UserCommand):
    action: Literal[Actions.SEARCH] = Actions.SEARCH
    k: int = Field(default=5, ge=1)
    prompt: str
    save_directory: Path = Field(default=Path("data/output/search_results"))
    chroma: bool = Field(default=False)
    expansion: bool = Field(default=False)


class SearchDatasetCommand(UserCommand):
    action: Literal[Actions.SEARCH_DATASET] = Actions.SEARCH_DATASET
    k: int = Field(default=5, ge=1)
    dataset_path: FilePath
    save_directory: Path = Field(default=Path("data/output/search_results"))
    chroma: bool = Field(default=False)
    expansion: bool = Field(default=False)


class AnswerCommand(UserCommand):
    action: Literal[Actions.ANSWER] = Actions.ANSWER
    prompt: str
    k: int = Field(default=1, ge=1)


class AnswerDatasetCommand(UserCommand):
    action: Literal[Actions.ANSWER_DATASET] = Actions.ANSWER_DATASET
    prompts_file: FilePath
    k: int = Field(default=1, ge=1)


class EvaluateCommand(UserCommand):
    action: Literal[Actions.EVALUATE] = Actions.EVALUATE
    answer_path: FilePath
    dataset_path: FilePath
    k: int = Field(default=1, ge=1)
    max_context_length: int = Field(default=2000, ge=1)


AnyCommand = Annotated[
    Union[IndexCommand, SearchCommand, SearchDatasetCommand,
          AnswerCommand, AnswerDatasetCommand, EvaluateCommand],
    Field(discriminator="action")
]
