from .models.CommandLine import (Actions,
                                 IndexCommand,
                                 SearchDatasetCommand,
                                 AnswerDatasetCommand,
                                 AnswerCommand,
                                 SearchCommand,
                                 EvaluateCommand)
from .parsing.parsing import parsing
from .index.index import Indexing
from .search.search import Search
from .answer.answer import Answer
from .search.searchDataset import SearchDataset
from .answer.answerDataset import AnswerDataset
from .evaluate.evaluate import Evaluate

from typing import cast


def main() -> None:
    try:
        userInput = parsing()
        match userInput.action:
            case Actions.INDEX:
                vllm: Indexing = Indexing(cast(IndexCommand, userInput))
                vllm.splitter()
                vllm.save_indexing()

            case Actions.SEARCH:
                userInput = cast(SearchCommand, userInput)
                search: Search = Search(
                    userInput.k,
                    userInput.prompt,
                    userInput.save_directory,
                    userInput.chroma,
                    questionid="q0",
                    expansion=userInput.expansion
                    )
                search.findMinimalSearchResults()
                search.findStudentSearchResults()
                search.saveStudentSearchResults()

            case Actions.SEARCH_DATASET:
                userInput = cast(SearchDatasetCommand, userInput)
                searchDataset: SearchDataset = SearchDataset(
                    k=userInput.k,
                    dataset_path=userInput.dataset_path,
                    output_dir=userInput.save_directory,
                    chroma=userInput.chroma,
                    expansion=userInput.expansion
                    )
                searchDataset.findAllQuestions()
                searchDataset.findQuestionsSources()
                searchDataset.saveSearchDataset()

            case Actions.ANSWER:
                userInput = cast(AnswerCommand, userInput)
                answer = Answer(userInput.prompt, userInput.k)
                answer.findSearchResult()
                answer.findChunks()
                answer.generate_answer()
                answer.createdAnswerFile()

            case Actions.ANSWER_DATASET:
                answerDataset = AnswerDataset(
                    cast(AnswerDatasetCommand, userInput)
                    )
                answerDataset.findSearchDatasetResult()
                answerDataset.createdAnswerDatasetFile()

            case Actions.EVALUATE:
                userInput = cast(EvaluateCommand, userInput)
                recall = Evaluate(userInput.dataset_path,
                                  userInput.k,
                                  userInput.answer_path,
                                  userInput.max_context_length)
                recall.calculate_recall()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
