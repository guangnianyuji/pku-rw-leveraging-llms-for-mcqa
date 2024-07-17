import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from utils import idx_to_ltr, prep_qwen_obj_for_save
from dataclasses import dataclass
import torch.nn.functional as F
import numpy as np
@dataclass
class ModelResponseMCP:
    mcp_logprobs: dict
    mcp_response_list: list


@dataclass
class ModelResponseCP:
    cp_logprobs: dict
    cp_unconditional_logprobs: dict
    cp_lens: dict
    cp_response_list: list


class Qwen2:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        """
        model = AutoModelForCausalLM.from_pretrained(
            "qwen/Qwen2-0.5B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen2-0.5B-Instruct")
        """
        self.model_name = "qwen/Qwen2-0.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                          torch_dtype="auto",
                                                          device_map=self.device).to("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.eval()
    
    def _get_response_for_cp(self, text):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        outputs = self.model(
            inputs.input_ids,            
        )      
        #print(type(outputs))
        logits=outputs.logits

        logits=logits[0]

        log_probs=[]

        for i in range(logits.shape[0]):
            score=logits[i]
            prob = F.softmax(score, dim=-1)
            log_prob = torch.log(prob[inputs.input_ids[0, i]])
            
            log_probs.append(log_prob.item())

        log_probs=np.array(log_probs)
        return {"logprobs": log_probs}
    
    def _get_response_for_mcp(self, text):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        num_return_sequences=5
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=1,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_scores=True,
        )
        # print("inputs.input_ids.shape",inputs.input_ids.shape)#torch.Size([1, 39])
        # print("sequence.shape",outputs.sequences.shape) #torch.Size([5, 40])
        # print("num_scores",outputs.scores[0].shape)#torch.Size([5, 151936])
        # outputs.scores[0].shape=torch.Size([5, 151936])
        scores=outputs.scores[0]#tuple的第一个
        log_probs={}
        
        for i in range(num_return_sequences):
            generated_ids = outputs.sequences[i]
            generated_ids = generated_ids[inputs.input_ids.shape[-1]:]
            score = scores[i]
            _log_probs = torch.log(score[generated_ids[0]])  #因为只输出一个字母
            _response_text=self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            log_probs[_response_text]=_log_probs.item()
            #print(f"_response_text==={_response_text}====")


        return {"logprobs": log_probs}


    def process_question_for_mcp(self, question):
        prompt_text = question.get_prompt_for_mcp()
        response_data = self._get_response_for_mcp(text=prompt_text)
        logprobs = response_data["logprobs"]

        return ModelResponseMCP(
            mcp_logprobs=logprobs,
            mcp_response_list=[
                prep_qwen_obj_for_save(
                    obj=response_data,
                    prompt_text=prompt_text
                )
            ]
        )

    def process_question_for_cp(self, question):
        prompt_text = question.get_prompt_for_cp()
        #print("prompt_text",prompt_text)

        response_list = list()
        logprobs = dict()
        unconditional_logprobs = dict()
        lens = dict()

        for idx, choice in enumerate(question.choices):
           # print("choice",choice)
            ltr = idx_to_ltr(idx)

            # 每个回答本身的概率
            response = self._get_response_for_cp(text=f"Answer: {choice}")
            choice_logprobs = response["logprobs"][2:-1]
            
            choice_n_tokens = len(choice_logprobs)
            unconditional_logprobs[ltr] = np.sum(choice_logprobs)
            #print("unconditional_logprobs",unconditional_logprobs[ltr])
            lens[ltr] = choice_n_tokens
            response_list.append(
                prep_qwen_obj_for_save(
                    obj=response,
                    prompt_text=f"Answer: {choice}"
                )
            )

            #以问题为条件的总概率
            response = self._get_response_for_cp(
                text=f"{prompt_text} {choice}"
            )
            token_logprobs = response["logprobs"]
            choice_logprobs = token_logprobs[-(choice_n_tokens+1):-1]

            logprobs[ltr] = np.sum(choice_logprobs)
            # print("以问题为条件的条件概率")
            # print("choice_logprobs",logprobs[ltr])
            # input("enter")
            response_list.append(
                prep_qwen_obj_for_save(
                    obj=response,
                    prompt_text=f"{prompt_text} {choice}"
                )
            )

        return ModelResponseCP(
            cp_logprobs=logprobs,
            cp_unconditional_logprobs=unconditional_logprobs,
            cp_lens=lens,
            cp_response_list=response_list
        )


if __name__ == "__main__":
    model=Qwen2()
    prompt="What is the capital of France?  B. London C. Berlin E. Madrid F. Paris"
    response_data=model._get_response_for_mcp(prompt)
