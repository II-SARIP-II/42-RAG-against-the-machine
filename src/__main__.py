from .models.CommandLine import (Actions,
                                 IndexCommand,
                                 SearchDatasetCommand,
                                 AnswerDatasetCommand,
                                 AnswerCommand,
                                 SearchCommand,
                                 EvaluateCommand)
from .parsing.parsing import parsing
from .index.vLLM import VllmIndexing
from .search.search import Search
from .answer.answer import Answer
from .search.searchDataset import SearchDataset
from .answer.answerDataset import AnswerDataset
from typing import cast
from .evaluate.evaluate import Evaluate


def main() -> None:
    userInput = parsing()
    match userInput.action:
        case Actions.INDEX:
            vllm: VllmIndexing = VllmIndexing(cast(IndexCommand, userInput))
            vllm.splitter()
        case Actions.SEARCH:
            userInput = cast(SearchCommand, userInput)
            search: Search = Search(
                userInput.k,
                userInput.prompt,
                userInput.save_directory,
                userInput.chroma,
                questionid="q0"
                )
            search.findMinimalSearchResults()
            search.findStudentSearchResults()
            search.saveStudentSearchResults()
        case Actions.SEARCH_DATASET:
            try:
                searchDataset: SearchDataset = SearchDataset(
                    cast(SearchDatasetCommand, userInput)
                    )
                searchDataset.findAllQuestions()
                searchDataset.findQuestionsSources()
                searchDataset.saveSearchDataset()
            except Exception as e:
                print(e)
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
            recall = Evaluate(cast(EvaluateCommand, userInput))
            recall.calculate_recall()
    print("WORK DONE !")


if __name__ == "__main__":
    main()
