from src.models.Result import DetailedSearchResults
from src.models.RagDataset import RagDataset
import json

# recall formule : Recall@k = (Number of queries with target in top-k) / (Total queries)


class Evaluate():
    def __init__(self, userCommand):
        self.config = userCommand

    def get_result(self, path_result):
        """Load search results JSON."""
        try:
            with (open(path_result, "r")as file):
                data = file.read()
                return DetailedSearchResults(**json.loads(data))
        except Exception:
            raise (ValueError(f"Cannot read {path_result}"))

    def get_answered_questions(self, path_answered_questions):
        """Load labeled questions JSON."""
        try:
            with (open(path_answered_questions, "r")as file):
                data = file.read()
                return RagDataset(**json.loads(data))
        except Exception:
            raise (ValueError(f"Cannot read {path_answered_questions}"))

    def calculate_recall():
        pass
