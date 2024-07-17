from typing import Callable
from dataclasses import dataclass
from prompts import QuestionPart

        
@dataclass
class DatasetInfo:
    path: str
    exemplar_split: str
    eval_split: str
    extractor: Callable
    name: str = None
    data_dir: str = None

def dataset_info():
    return {"ARC-Challenge": DatasetInfo(
        path="allenai/ai2_arc",
        name="ARC-Challenge",
        exemplar_split="train",
        eval_split="validation",
        extractor=lambda row: {
            "parts": [
                QuestionPart(text=row["question"], tag="Question")
            ],
            "choices": row["choices"]["text"],
            "answer_idx": row["choices"]["label"].index(row["answerKey"])
        }
    ),
    "ARC-Challenge-test": DatasetInfo(
        path="allenai/ai2_arc",
        name="ARC-Challenge",
        exemplar_split="train",
        eval_split="test",
        extractor=lambda row: {
            "parts": [
                QuestionPart(text=row["question"], tag="Question")
            ],
            "choices": row["choices"]["text"],
            "answer_idx": row["choices"]["label"].index(row["answerKey"])
        }
    ),
    }

"""

{
"answerKey": "B",
"choices": {
    "label": ["A", "B", "C", "D"],
    "text": ["Shady areas increased.", "Food sources increased.", "Oxygen levels increased.", "Available water increased."]
},
"id": "Mercury_SC_405487",
"question": "One year, the oak trees in a park began producing more acorns than usual. The next year, the population of chipmunks in the park also increased. Which best explains why there were more chipmunks the next year?"
}

"""