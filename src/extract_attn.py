import argparse
import os
import torch
import transformers
from transformers import AutoTokenizer

from constants import Constants
from prompts import Prompts

parser = argparse.ArgumentParser()
parser.add_argument("--model_size", type=str, default="large")
parser.add_argument("--prompt_difficulty", type=str, default="medium")
parser.add_argument("--prompt_category", type=str, default=None)
parser.add_argument("--prompt_n_shots", type=int, default=None)
args = parser.parse_args()


def extract_attention(model, tokenizer, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model_size == 'small':
        model = transformers.AutoModelForCausalLM.from_pretrained(
            Constants.SMALL_MODEL_NAME_CAUSAL_LM,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(Constants.SMALL_MODEL_NAME_CAUSAL_LM)
    elif args.model_size == 'large':
        model = transformers.AutoModelForCausalLM.from_pretrained(
            Constants.LARGE_MODEL_NAME_CAUSAL_LM,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(Constants.LARGE_MODEL_NAME_CAUSAL_LM)
    else:
        raise ValueError(f"Invalid model choice: {args.model_size}")

    git_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    prompts = Prompts(os.path.join(git_root, "prompts.json"))       
    prompt = prompts.get_prompt("TASK QUERY", difficulty=args.prompt_difficulty, category=args.prompt_category, n_shots=args.prompt_n_shots)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Replace the generate call with forward pass to get attention values
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # attention_values will be a tuple of attention tensors for each layer
    # Each tensor has shape (batch_size, num_heads, sequence_length, sequence_length)
    attention_values = outputs.attentions


    # Convert attention values to numpy and save
    import numpy as np

    attention_arrays = [attn.cpu().to(torch.float32).numpy() for attn in attention_values]
    np.save('attention_values.npy', attention_arrays) 