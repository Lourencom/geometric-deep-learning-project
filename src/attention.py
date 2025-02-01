import argparse
import os
import torch
import numpy as np

import transformers 
from transformers import AutoTokenizer

from utils import get_git_root, get_model_and_tokenizer
from constants import Constants
from prompts import Prompts

def extract_attention(args):
    model, tokenizer = get_model_and_tokenizer(args.model_size)

    prompts = Prompts(args.prompt_path)       
    prompt = prompts.get_prompt(
        difficulty=args.prompt_difficulty,
        category=args.prompt_category,
        n_shots=args.prompt_n_shots
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attention_values = outputs.attentions

    attention_arrays = [attn.cpu().to(torch.float32).numpy() for attn in attention_values]

    git_root_path = get_git_root()
    output_dir = os.path.join(git_root_path, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)  
    np.save(os.path.join(output_dir, f"attention_values_{args.model_size}_{args.prompt_difficulty}_{args.prompt_category}_{args.prompt_n_shots}.npy"), attention_arrays) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="large")
    parser.add_argument("--prompt_path", type=str, default="data/prompts.json")
    parser.add_argument("--prompt_difficulty", type=str, default="medium")
    parser.add_argument("--prompt_category", type=str, default=None)
    parser.add_argument("--prompt_n_shots", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="data/attn")
    args = parser.parse_args()

    extract_attention(args)