from datasets import load_dataset
from config import SEED
import random
from prompts import Exemplar, QuestionPart, QuestionWithExemplars

def get_questions_with_exemplars(info,n_shots):
    examplar_dataset=load_dataset(info.path,info.name,split=info.exemplar_split)

    exemplars = [Exemplar(**info.extractor(row)) for row in examplar_dataset]

    eval_dataset=load_dataset(info.path,info.name,split=info.eval_split)

    random.seed(SEED)

    quesions_with_examplars=list()

    for idx,row in enumerate(eval_dataset):
        possible_idxs = list(range(len(exemplars)))
        row_exemplars = [
            exemplars[i] for i in random.sample(possible_idxs, n_shots)
        ]

        row_quesions_with_examplars=QuestionWithExemplars(
            parts=[QuestionPart(text=row["question"],tag="Question")],
            choices=row["choices"]["text"],
            answer_idx=row["choices"]["label"].index(row["answerKey"]),
            exemplars=row_exemplars
        )

        quesions_with_examplars.append(row_quesions_with_examplars)

    return quesions_with_examplars


    






    
    