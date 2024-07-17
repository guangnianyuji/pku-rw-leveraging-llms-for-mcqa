import torch

def idx_to_ltr(idx):
    return chr(idx + ord("A"))


def ltr_to_idx(ltr):
    return ord(ltr) - ord("A")


def prep_qwen_obj_for_save(obj, prompt_text=None):
    # 如果 obj 是 transformers 的 output 对象
    if isinstance(obj, dict):
        for key in obj.keys():
            if isinstance(obj[key], torch.Tensor):
                obj[key] = obj[key].cpu().tolist()  # 将 tensor 转为列表，便于保存
            elif isinstance(obj[key], list):
                for i in range(len(obj[key])):
                    if isinstance(obj[key][i], torch.Tensor):
                        obj[key][i] = obj[key][i].cpu().tolist()  # 将 tensor 转为列表
                    elif isinstance(obj[key][i], dict):
                        obj[key][i] = prep_qwen_obj_for_save(obj[key][i])  # 递归调用处理字典
            elif isinstance(obj[key], dict):
                obj[key] = prep_qwen_obj_for_save(obj[key])  # 递归调用处理字典
    if prompt_text is not None:
        obj["prompt_text"] = prompt_text
    return obj
