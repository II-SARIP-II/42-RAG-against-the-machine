from .models.CommandLine import Actions, IndexCommand, SearchDatasetCommand, AnswerDatasetCommand
from .parsing.parsing import parsing
from .index.vLLM import VllmIndexing
from .search.search import Search
from .answer.answer import Answer
from .search.searchDataset import SearchDataset
from .answer.answerDataset import AnswerDataset
from typing import cast


def main() -> None:
    userInput = parsing()
    match userInput.action:
        case Actions.INDEX:
            vllm: VllmIndexing = VllmIndexing(cast(IndexCommand, userInput))
            vllm.splitter()
        case Actions.SEARCH:
            search: Search = Search(userInput.k, userInput.prompt, userInput.save_directory, userInput.chroma)
            search.findMinimalSearchResults()
            search.findStudentSearchResults()
            search.saveStudentSearchResults()
        case Actions.SEARCH_DATASET:
            try:

                search: SearchDataset = SearchDataset(cast(SearchDatasetCommand, userInput))
                search.findAllQuestions()
                search.findQuestionsSources()
                search.saveSearchDataset()
            except Exception as e:
                print(e)
        case Actions.ANSWER:
            answer = Answer(userInput.prompt, userInput.k)
            answer.findSearchResult()
            answer.findChunks()
            answer.generate_answer()
            answer.createdAnswerFile()
        case Actions.ANSWER_DATASET:
            answerDataset = AnswerDataset(cast(AnswerDatasetCommand, userInput))
            answerDataset.findSearchDatasetResult()
            answerDataset.createdAnswerDatasetFile()
        case Actions.EVALUATE:
            pass
    print("WORK DONE !")


if __name__ == "__main__":
    main()
