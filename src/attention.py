import argparse
import os
import torch
import numpy as np
from utils import relative_to_absolute_path


def extract_attention(args, outputs, save=False):
    """
    Returns an array of attention matrices.

    For the prompt, we have x layers.
    We return an array of shape (x, b, a, n, n) where b is the batch size, a is the number of attention heads, n is the number of tokens in the prompt.

    For the intermediate steps, we have the same but we have y intermediate steps.
    We return an array of shape (y, x, b, a, n, n).

    """
    prompt_outputs, intermediate_outputs = outputs

    attention_values = prompt_outputs.attentions
    intermediate_attention_values = intermediate_outputs.attentions

    attention_arrays = np.array([attn.cpu().to(torch.float32).numpy().squeeze() for attn in attention_values])
    intermediate_attention_arrays = []
    for i, attn_step in enumerate(intermediate_attention_values):
        indermediate_attn_array = np.array([attn.cpu().to(torch.float32).numpy().squeeze() for attn in attn_step])
        intermediate_attention_arrays.append(indermediate_attn_array)

    output_dir = relative_to_absolute_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if save:
        # FIXME: doesnt work for now
        prompt_attns_filename = os.path.join(output_dir, f"attention_values_{args.model_size}_{args.prompt_difficulty}_{args.prompt_category}_{args.prompt_n_shots}.npy")
        intermediate_attns_filename = os.path.join(output_dir, f"intermediate_attention_values_{args.model_size}_{args.prompt_difficulty}_{args.prompt_category}_{args.prompt_n_shots}.npy")
        np.save(prompt_attns_filename, attention_arrays, allow_pickle=True) 
        np.save(intermediate_attns_filename, intermediate_attention_arrays, allow_pickle=True)
    return {"prompt_attns": attention_arrays, "intermediate_attns": intermediate_attention_arrays}
