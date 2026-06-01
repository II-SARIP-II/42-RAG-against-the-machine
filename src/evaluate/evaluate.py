from src.models.Result import (StudentDetailedSearchResults,
                               DetailedSearchResults)
from src.models.RagDataset import RagDataset
import json
from src.models.CommandLine import EvaluateCommand
from src.models.Question import AnsweredQuestion, UnansweredQuestion
from src.models.Source import DetailedSource, MinimalSource


class Evaluate():
    def __init__(self, userCommand: EvaluateCommand) -> None:
        self.config = userCommand
        self.searched_data = self.get_result()
        self.answers = self.get_answered_questions()

    def get_result(self) -> StudentDetailedSearchResults:
        """Load search results JSON."""
        try:
            print(self.config.dataset_path)
            with open(self.config.dataset_path, "r", encoding="utf-8") as file:
                data = file.read()
                return StudentDetailedSearchResults(**json.loads(data))
        except Exception:
            raise (ValueError(f"Cannot read {self.config.dataset_path}"))

    def get_answered_questions(self) -> RagDataset:
        """Load labeled questions JSON."""
        try:
            with open(self.config.answer_path, "r", encoding="utf-8") as file:
                data = file.read()
                return RagDataset(**json.loads(data))
        except Exception:
            raise (ValueError(f"Cannot read {self.config.answer_path}"))

    def calculate_recall(self) -> float:
        total_expected = len(self.answers.rag_questions)
        print(total_expected, len(self.searched_data.search_results))
        if total_expected == 0:
            print("Recall: 0.0%")
            return 0.0

        answers_dict = {answer.question_id: answer
                        for answer in self.answers.rag_questions}
        count = 0

        for question in self.searched_data.search_results:
            if question.question_id in answers_dict:
                if self.is_matching(question, answers_dict):
                    count += 1
                    print(question.question)
        recall = (count / total_expected) * 100
        print(f"Recall: {recall:.2f}%")
        return recall

    def is_matching(self,
                    question: DetailedSearchResults,
                    answers_dict: dict[str, AnsweredQuestion |
                                       UnansweredQuestion]
                    ) -> bool:
        expected_answer = answers_dict[question.question_id]
        if not isinstance(expected_answer, AnsweredQuestion):
            return False

        for qsource in question.retrieved_sources:
            for asource in expected_answer.sources:
                if qsource.file_path != asource.file_path:
                    continue
                if not self.is_overlaped(qsource, asource):
                    continue
                return True
        return False

    def is_overlaped(
            self,
            qsource: DetailedSource,
            asource: MinimalSource
            ) -> bool:
        overlap_start = max([qsource.first_character_index,
                             asource.first_character_index])
        overlap_end = min([qsource.last_character_index,
                           asource.last_character_index])
        overlap_len = overlap_end - overlap_start
        if overlap_len < 0:
            return False
        asrc_len = asource.last_character_index - asource.first_character_index
        overlap_score = (overlap_len / asrc_len)
        return overlap_score >= 0.05
