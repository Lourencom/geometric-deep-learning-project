import argparse
import os
import torch
import numpy as np
from model import run_model, get_tokenwise_attns
from prompts import filter_prompts, Prompts

def aggregate_attention_layers(attn_matrices):
    """
    Aggregates a list of attention matrices using matrix multiplication that represents the overall information flow across layers.
    Example:
        For two layers, the aggregated attention is computed as:
            A_agg = A_layer2 @ A_layer1
    """
    # Start with the first layer's attention matrix.
    A_agg = attn_matrices[0]
    # Multiply successively by the next layer's attention matrix.
    for attn in attn_matrices[1:]:
        A_agg = np.dot(attn, A_agg)
    return np.clip(A_agg, 0, 1) # sometimes numerical errors lead to values slightly outside [0, 1]



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
        intermediate_attn_array = np.array([attn.cpu().to(torch.float32).numpy().squeeze() for attn in attn_step])
        intermediate_attention_arrays.append(intermediate_attn_array)

    if save:
        family, size, variant = args.current_model
        model_identifier = f"{family}_{size}_{variant}"
        prompt_attns_filename = os.path.join(output_dir, 
            f"prompt_attention_values_{model_identifier}_{args.prompt_id}.npy")
        np.save(prompt_attns_filename, attention_arrays, allow_pickle=True)
        
    return {"prompt_attns": attention_arrays, "intermediate_attns": intermediate_attention_arrays}


def get_cached_attention(args, attn_dir, model_tuple, prompt_id):
    family, size, variant = model_tuple
    model_identifier = f"{family}_{size}_{variant}"
    
    cached_attention_files = [el for el in os.listdir(attn_dir)
                        if "prompt_attention_values" in el and el.endswith(".npy") 
                        and f"_{prompt_id}" in el and model_identifier in el]
    
    cached_attentions = filter_prompts(cached_attention_files, args.prompt_difficulty, args.prompt_category, args.prompt_n_shots, model_identifier)
    return cached_attentions


def load_attns(args, models=None, save=False, **kwargs):
    if args.analysis_type == "tokenwise":
        return load_attn_tokenwise(args, models, save, **kwargs)
    elif args.analysis_type == "layerwise":
        return load_attn_layerwise(args, models, save, **kwargs)
    else:
        raise ValueError(f"Invalid analysis type: {args.analysis_type}")

def load_attn_layerwise(args, models=None, save=False, **kwargs):
    attn_dicts = []
    models_to_process = models if models is not None else args.models
    
    for model_tuple in models_to_process:
        attn_path = kwargs.get("attn_dir", args.attn_dir)
        cached_attentions = get_cached_attention(args, attn_path, model_tuple, args.prompt_id)
        
        if len(cached_attentions) == 0:
            args.current_model = model_tuple  # Set current model for processing
            outputs, *_ = run_model(args)
            attn_dict = extract_attention(args, outputs, attn_path, save=save)["prompt_attns"]
        else:
            attn_dict = np.load(os.path.join(attn_path, cached_attentions[0]), allow_pickle=True)
        attn_dicts.append(attn_dict)
    return attn_dicts



def load_attn_tokenwise(args, models=None, save=False, **kwargs):
    attn_dicts = []
    models_to_process = models if models is not None else args.models
    
    for model_tuple in models_to_process:
        current_model = model_tuple  # Set current model for processing
        prompts = Prompts(args.prompt_path)       
        prompt = prompts.get_prompt(
            prompt_id=args.prompt_id,
            difficulty=args.prompt_difficulty,
            category=args.prompt_category,
            n_shots=args.prompt_n_shots
        )['prompt']
        attn_dict = get_tokenwise_attns(current_model, prompt)
        attn_dicts.append(attn_dict)
    return [el['attention_matrices'] for el in attn_dicts]