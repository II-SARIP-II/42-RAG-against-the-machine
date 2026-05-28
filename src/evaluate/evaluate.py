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
            with (open(self.config.dataset_path, "r")as file):
                data = file.read()
                return StudentDetailedSearchResults(**json.loads(data))
        except Exception:
            raise (ValueError(f"Cannot read {self.config.dataset_path}"))

    def get_answered_questions(self):
        """Load labeled questions JSON."""
        try:
            with (open(self.config.answer_path, "r")as file):
                data = file.read()
                return RagDataset(**json.loads(data))
        except Exception:
            raise (ValueError(f"Cannot read {self.config.answer_path}"))

    def calculate_recall(self):
        print(self.searched_data, self.answers)
