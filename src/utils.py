import subprocess
import os
import transformers
import torch
from transformers import AutoTokenizer

from constants import Constants

def get_git_root():
    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
        ).strip().decode("utf-8")
        return git_root
    except subprocess.CalledProcessError:
        return None  # Not inside a Git repository
    
def relative_to_absolute_path(path):
    return os.path.join(get_git_root(), path)
    
def get_model_and_tokenizer(model_size):
    if model_size == "small":
        model = transformers.AutoModelForCausalLM.from_pretrained(
            Constants.SMALL_MODEL_NAME_CAUSAL_LM,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(Constants.SMALL_MODEL_NAME_CAUSAL_LM)
    
    elif model_size == 'large':
        model = transformers.AutoModelForCausalLM.from_pretrained(
            Constants.LARGE_MODEL_NAME_CAUSAL_LM,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(Constants.LARGE_MODEL_NAME_CAUSAL_LM)
    
    elif model_size == 'huge':
        model = transformers.AutoModelForCausalLM.from_pretrained(
            Constants.HUGE_MODEL_NAME_INSTRUCT,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(Constants.HUGE_MODEL_NAME_INSTRUCT)
        
    else:
        raise ValueError(f"Invalid model size: {model_size}")
    return model, tokenizer


def filter_prompts(prompts, prompt_difficulty, prompt_category, prompt_n_shots, model_size):
    if prompt_difficulty is not None:
        prompts = [p for p in prompts if prompt_difficulty in p]
    if prompt_category is not None:
        prompts = [p for p in prompts if prompt_category in p]
    if prompt_n_shots is not None:
        prompts = [p for p in prompts if prompt_n_shots in p]
    if model_size is not None:
        prompts = [p for p in prompts if model_size in p]
    return prompts