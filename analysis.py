from collections import defaultdict
from utils import idx_to_ltr
import pandas as pd


class ExperimentSaver(defaultdict):

    def __init__(self, save_fname):
        super().__init__(list)
        self.save_fname = save_fname

    def save(self):
        print("Saving to", self.save_fname)
        pd.DataFrame(self).to_pickle(self.save_fname)

def add_correct_answer_col(df):
    df["correct_answer"] = df.apply(
        lambda row: idx_to_ltr(row["qwe"]["answer_idx"]),
        axis=1
    )
def div_dicts(a, b):
    new_dict = dict()
    for key in a.keys():
        if key in b.keys():
            new_dict[key] = a[key] / b[key]
    return new_dict

def sub_dicts(a, b):
    new_dict = dict()
    for key in a.keys():
        if key in b.keys():
            new_dict[key] = a[key] - b[key]
    return new_dict

def analysis_result(fname):

    # Load file
    df = pd.read_pickle(fname)

    # Add correct answer column
    df["correct_answer"] = df.apply(
    lambda row: idx_to_ltr(row["qwe"]["answer_idx"]),
    axis=1
    )
    df["chosen_answer_MCP"] = df.apply(
                lambda row: max(
                    row["model_response_mcp"]["mcp_logprobs"].items(),
                    key=lambda x: x[1]
                    # In line below [0] is the key (as opposed to value)
                    # Additionally we use 1: instead of lstrip because
                    # we want the prediction "A" to be wrong when " A"
                    # is expected, for example
                )[0],
                axis=1
            )
    df["correct_MCP"] = df.apply(
        lambda row: row["chosen_answer_MCP"] == row["correct_answer"],
        axis=1
    )
    print(
        "Accuracy MCP:",
        df["correct_MCP"].mean()
    )

    df["chosen_answer_RAW"] = df.apply(
        lambda row: max(
            row["model_response_cp"]["cp_logprobs"].items(),
            key=lambda x: x[1]
            # In line below [0] is the key (as opposed to value)
            # Additionally we use 1: instead of lstrip because
            # we want the prediction "A" to be wrong when " A"
            # is expected, for example
        )[0],
        axis=1
    )
    df["correct_RAW"] = df.apply(
        lambda row: row["chosen_answer_RAW"] == row["correct_answer"],
        axis=1
    )
    print(
        "Accuracy RAW:",
        df["correct_RAW"].mean()
    )

    df["chosen_answer_LN"] = df.apply(
        lambda row: max(
            div_dicts(
                row["model_response_cp"]["cp_logprobs"],
                row["model_response_cp"]["cp_lens"]
            ).items(),
            key=lambda x: x[1]
        )[0],
        axis=1
    )

    df["correct_LN"] = df.apply(
        lambda row: row["chosen_answer_LN"] == row["correct_answer"],
        axis=1
    )
    print(
        "Accuracy LN:",
        df["correct_LN"].mean()
    )

    df["chosen_answer_UN"]=df.apply(
        lambda row: max(
            sub_dicts(
                row["model_response_cp"]["cp_logprobs"],
                row["model_response_cp"]["cp_unconditional_logprobs"]
            ).items(),
            key=lambda x: x[1]
        )[0],
        axis=1
    )
    df["correct_UN"] = df.apply(
        lambda row: row["chosen_answer_UN"] == row["correct_answer"],
        axis=1
    )
    print(
        "Accuracy UN:",
        df["correct_UN"].mean()
    )

    result_txt=fname.replace(".pkl",".txt")
    with open(result_txt,"w") as f:
        f.write("Accuracy MCP: "+str(df["correct_MCP"].mean())+"\n")
        f.write("Accuracy RAW: "+str(df["correct_RAW"].mean())+"\n")
        f.write("Accuracy LN: "+str(df["correct_LN"].mean())+"\n")
        f.write("Accuracy UN: "+str(df["correct_UN"].mean())+"\n")
        f.write("Results saved to: "+result_txt)

