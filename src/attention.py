import os
import torch
import numpy as np
from prompts import filter_prompts, Prompts
from constants import get_model_and_tokenizer
from model_utils import sample_next_token

def aggregate_attention_layers(attn_matrices):
    """
    Aggregates a list of attention matrices.

    Uses attention rollout. https://arxiv.org/abs/2005.00928

    """
    discard_ratio = 0.0
    # init rollout with identity matrix
    rollout = np.eye(attn_matrices[0].shape[0])

    for layer_idx, attn in enumerate(attn_matrices):
        # optionally discard certai ratio of lowest attn values
        if discard_ratio > 0:
            flat = attn.flatten()
            threshold = np.quantile(flat, discard_ratio)
            attn = np.where(attn < threshold, 0, attn)
        
        # Add the residual connection by summing with the identity matrix
        attn_with_residual = attn + np.eye(attn.shape[0])
        
        # Normalize each row so that the attention weights sum to 1
        attn_norm = attn_with_residual / attn_with_residual.sum(axis=-1, keepdims=True)
        
        # Multiply the normalized attention with the cumulative rollout.
        # Note: Using matrix multiplication (@) such that the order reflects the
        # sequential propagation of attention from the first to the last layer.
        rollout = attn_norm @ rollout
    
    return rollout


def extract_attention(current_model, prompt_id, outputs, output_dir, save=False):
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
        family, size, variant = current_model
        model_identifier = f"{family}_{size}_{variant}"
        prompt_attns_filename = os.path.join(output_dir, 
            f"prompt_attention_values_{model_identifier}_{prompt_id}.npy")
        np.save(prompt_attns_filename, attention_arrays, allow_pickle=True)
        
    return {"prompt_attns": attention_arrays, "intermediate_attns": intermediate_attention_arrays}


def get_token_by_token_attention(model, tokenizer, prompt_text, max_new_tokens=512):
    """
    Captures the full attention matrices for each generated token across all layers.
    
    This function performs token-by-token generation and captures the attention matrices
    at each step, preserving the full attention state (all heads, all layers) for each
    token generation step.
    
    Args:
        model: The HuggingFace model to use for generation
        tokenizer: The tokenizer corresponding to the model
        prompt_text: The input prompt text
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        A dictionary containing:
        - 'prompt_tokens': The tokenized prompt
        - 'generated_tokens': The generated tokens
        - 'full_text': The complete text (prompt + generation)
        - 'attention_matrices': A list of attention matrices for each generation step
          Each element is a list of layer attentions, where each layer attention is
          a tensor of shape [batch_size, num_heads, curr_seq_len, curr_seq_len]
        - 'token_ids': The token IDs for each step of generation
        - 'token_texts': The decoded text for each generated token
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    input_length = input_ids.shape[1]
    
    # Store the results
    attention_matrices = []
    generated_token_ids = []
    token_texts = []
    
    # Start with the prompt tokens
    current_ids = input_ids.clone()
    
    # Generate tokens one by one
    with torch.no_grad():
        # First get prompt attention
        prompt_outputs = model(current_ids, output_attentions=True, return_dict=True)
        prompt_attentions = prompt_outputs.attentions
        
        # Then generate tokens
        for _ in range(max_new_tokens):
            # Forward pass with attention outputs
            outputs = model(current_ids, output_attentions=True, return_dict=True)
            
            # Get the next token prediction
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = sample_next_token(current_ids, next_token_logits)
            
            # Store the attention matrices for this step
            # This captures attention for all layers and all heads
            # Each attention matrix is of shape [batch_size, num_heads, curr_seq_len, curr_seq_len]
            step_attentions = outputs.attentions
            attention_matrices.append(step_attentions)
            
            # Store the generated token
            generated_token_ids.append(next_token_id.item())
            token_text = tokenizer.decode(next_token_id.item())
            token_texts.append(token_text)
            
            # Check if we've hit the end of sequence token
            if next_token_id.item() == tokenizer.eos_token_id:
                break
                
            # Append the new token to the sequence
            current_ids = torch.cat([current_ids, next_token_id], dim=1)
    
    # Decode the full generated text
    full_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
    generated_text = tokenizer.decode(current_ids[0][input_length:], skip_special_tokens=True)
    
    return {
        'prompt_tokens': input_ids,
        'prompt_attentions': prompt_attentions,
        'generated_tokens': generated_token_ids,
        'full_text': full_text,
        'generated_text': generated_text,
        'attention_matrices': attention_matrices,
        'token_texts': token_texts,
        'prompt_text': prompt_text,
    }


def get_tokenwise_attns(current_model, prompt="Explain the concept of attention in transformer models."):
    model_family, model_size, model_variant = current_model
    model, tokenizer = get_model_and_tokenizer(model_family, model_size, model_variant)
    attention_data = get_token_by_token_attention(model, tokenizer, prompt, max_new_tokens=50)        
    return attention_data

def get_cached_attention(attn_dir, model_tuple, prompt_id, prompt_difficulty, prompt_category, prompt_n_shots):
    family, size, variant = model_tuple
    model_identifier = f"{family}_{size}_{variant}"
    
    cached_attention_files = [el for el in os.listdir(attn_dir)
                        if "prompt_attention_values" in el and el.endswith(".npy") 
                        and f"_{prompt_id}" in el and model_identifier in el]
    
    cached_attentions = filter_prompts(cached_attention_files, prompt_difficulty, prompt_category, prompt_n_shots, model_identifier)
    return cached_attentions


def load_attns(args, models=None, save=False, **kwargs):
    if args.analysis_type == "tokenwise":
        return load_attn_tokenwise(args, models, save, **kwargs)
    else:
        raise ValueError(f"Invalid analysis type: {args.analysis_type}")


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




