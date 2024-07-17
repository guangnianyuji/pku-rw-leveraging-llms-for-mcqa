from argparse import ArgumentParser

def get_config_from_args():
    parser = ArgumentParser()
    parser.add_argument("--ds_name", help="Dataset name",default="ARC-Challenge")
    parser.add_argument("--model_name", help="Style name",default="Qwen2")
    parser.add_argument("--n_shots", type=int, help="# of shots",default=5)
    parser.add_argument("--seed", type=int, help="Random seed",default=42)
    
    args = parser.parse_args()
    return args

