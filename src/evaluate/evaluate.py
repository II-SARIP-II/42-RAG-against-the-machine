from src.models.Result import StudentDetailedSearchResults
from src.models.RagDataset import RagDataset
import json

# recall formule : Recall@k = (Number of queries with target in top-k) / (Total queries)


class Evaluate():
    def __init__(self, userCommand):
        self.config = userCommand
        self.searched_data = self.get_result()
        self.answers = self.get_answered_questions()

    def get_result(self):
        """Load search results JSON."""
        try:
            print(self.config.dataset_path)
            with open(self.config.dataset_path, "r", encoding="utf-8") as file:
                data = file.read()
                return StudentDetailedSearchResults(**json.loads(data))
        except Exception:
            raise (ValueError(f"Cannot read {self.config.dataset_path}"))

    def get_answered_questions(self):
        """Load labeled questions JSON."""
        try:
            with open(self.config.answer_path, "r", encoding="utf-8") as file:
                data = file.read()
                return RagDataset(**json.loads(data))
        except Exception:
            raise (ValueError(f"Cannot read {self.config.answer_path}"))

    def calculate_recall(self):
        answers_dict = {answer.question_id: answer for answer in self.answers.rag_questions}
        count = 0
        for question in self.searched_data.search_results:
            if question.question_id in answers_dict:

                

    def is_matching()
        for qsource in question.retrieved_sources:
            if qsource.file_path != answers_dict[question.question_id].sources.file_path:
                continue
            if not is_overlaped(qsource, answers_dict[question.question_id].sources):
                continue
            return True
        return False