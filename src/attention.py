import argparse
import os
import torch
import numpy as np

import transformers 
from transformers import AutoTokenizer

from utils import get_git_root, get_model_and_tokenizer
from constants import Constants
from prompts import Prompts

def run_model(args):
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

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return outputs, answer, prompt


def extract_attention(args, outputs):
    attention_values = outputs.attentions

    attention_arrays = [attn.cpu().to(torch.float32).numpy() for attn in attention_values]

    git_root_path = get_git_root()
    output_dir = os.path.join(git_root_path, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)  
    np.save(os.path.join(output_dir, f"attention_values_{args.model_size}_{args.prompt_difficulty}_{args.prompt_category}_{args.prompt_n_shots}.npy"), attention_arrays) 


def store_answer(args, answer, prompt):
    git_root_path = get_git_root()
    output_dir = os.path.join(git_root_path, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the prompt and the answer to a text file for reference
    answer_file = os.path.join(output_dir, f"generated_answer_{args.model_size}_{args.prompt_difficulty}_{args.prompt_category}_{args.prompt_n_shots}.txt")
    with open(answer_file, "w") as f:
        f.write("Prompt:\n")
        f.write(prompt + "\n\n")
        f.write("Generated Answer:\n")
        f.write(answer)
    return answer_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="large")
    parser.add_argument("--prompt_path", type=str, default="data/prompts.json")
    parser.add_argument("--prompt_difficulty", type=str, default="medium")
    parser.add_argument("--prompt_category", type=str, default=None)
    parser.add_argument("--prompt_n_shots", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="data/attn")
    args = parser.parse_args()

    outputs, answer, prompt = run_model(args)
    extract_attention(args, outputs)
    store_answer(args, answer, prompt)
