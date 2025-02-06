import argparse
import os
import torch
import numpy as np
from utils import get_git_root

def extract_attention(args, outputs):
    attention_values = outputs.attentions

    attention_arrays = [attn.cpu().to(torch.float32).numpy() for attn in attention_values]

    git_root_path = get_git_root()
    output_dir = os.path.join(git_root_path, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)  
    np.save(os.path.join(output_dir, f"attention_values_{args.model_size}_{args.prompt_difficulty}_{args.prompt_category}_{args.prompt_n_shots}.npy"), attention_arrays) 
    return attention_arrays

def load_attention(file_path):
    return np.load(file_path)
