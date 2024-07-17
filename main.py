from modelscope import AutoModelForCausalLM, AutoTokenizer
from dataset_info import dataset_info
from get_questions_from_dataset import get_questions_with_exemplars
from models import Qwen2
from arguments import get_config_from_args
from analysis import ExperimentSaver,analysis_result
from tqdm import tqdm
import os
def main(args):

    model=Qwen2()

    info=dataset_info()[args.ds_name]

    questions_with_exemplars=get_questions_with_exemplars(info,args.n_shots)

    save_fname="_".join([args.ds_name,args.model_name,str(args.n_shots),str(args.seed)])
    dir="results/"+save_fname
    if not os.path.exists(dir):
        os.makedirs(dir,exist_ok=True)
    save_fname="results/"+save_fname+"/result.pkl"

    saver = ExperimentSaver(save_fname=save_fname)

    for qwe in tqdm(questions_with_exemplars):
       #print("answer_idx",qwe.answer_idx)
        response_cp=model.process_question_for_cp(qwe)
        response_mcp=model.process_question_for_mcp(qwe)
        
        saver["qwe"].append(vars(qwe))
        saver["model_response_mcp"].append(vars(response_mcp))
        saver["model_response_cp"].append(vars(response_cp))
    saver.save()
    analysis_result(save_fname)



    

if __name__ == "__main__":
    args=get_config_from_args()
    main(args)