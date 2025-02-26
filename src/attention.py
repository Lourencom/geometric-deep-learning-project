import argparse
import os
import torch
import numpy as np
from model import run_model
from utils import filter_prompts

def extract_attention(args, outputs, output_dir, save=False):
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

    if save:
        # FIXME: doesnt work for now
        prompt_attns_filename = os.path.join(output_dir, f"prompt_attention_values_{args.model_size}_{args.prompt_id}.npy")
        #intermediate_attns_filename = os.path.join(output_dir, f"intermediate_attention_values_{args.model_size}_{args.prompt_id}.npy")
        np.save(prompt_attns_filename, attention_arrays, allow_pickle=True) 
        #np.save(intermediate_attns_filename, np.array(intermediate_attention_arrays, dtype=object), allow_pickle=True)
    return {"prompt_attns": attention_arrays, "intermediate_attns": intermediate_attention_arrays}


def aggregate_attention_layers(attn_matrices):
    """
    Aggregates a list of attention matrices using matrix multiplication.
    
    Parameters:
        attn_matrices (list of np.ndarray): 
            List of attention matrices where each matrix is of shape (n_tokens, n_tokens).
            The list should be ordered from the first layer to the last layer.
            
    Returns:
        np.ndarray: Aggregated attention matrix of shape (n_tokens, n_tokens) that 
                    represents the overall information flow across layers.
                    
    Example:
        For two layers, the aggregated attention is computed as:
            A_agg = A_layer2 @ A_layer1
    """
    # Start with the first layer's attention matrix.
    A_agg = attn_matrices[0]
    # Multiply successively by the next layer's attention matrix.
    for attn in attn_matrices[1:]:
        A_agg = np.dot(attn, A_agg)
    return A_agg


def get_cached_attention(args, attn_dir, model_size):
    cached_attention_files = [el for el in os.listdir(attn_dir)
                        if "prompt_attention_values" in el and el.endswith(".npy")]
    
    cached_attentions = filter_prompts(cached_attention_files, args.prompt_difficulty, args.prompt_category, args.prompt_n_shots, model_size)
    
    return cached_attentions


def load_attns(args, model_sizes=["small", "large"], save=False, **kwargs):
    # FIXME: this currently does not return a dict, returns the numpy arrays directly
    attn_dicts = []
    for model_size in model_sizes:
        attn_path = kwargs.get("attn_dir", args.attn_dir)
        cached_attentions = get_cached_attention(args, attn_path, model_size)
        if len(cached_attentions) == 0:
            args.model_size = model_size
            outputs, *_ = run_model(args)
            attn_dict = extract_attention(args, outputs, attn_path, save=save)["prompt_attns"]
        else:
            attn_dict = np.load(os.path.join(attn_path, cached_attentions[0]), allow_pickle=True)
        attn_dicts.append(attn_dict)
    return attn_dicts
