import os
import torch
import numpy as np
from prompts import filter_prompts, Prompts
from constants import get_model_and_tokenizer
from model_utils import sample_next_token


def aggregate_attention_layers(attn_matrices, alpha = 0.8):
    """
    Aggregates a list of attention matrices.

    Uses attention rollout. https://arxiv.org/abs/2005.00928


    WARNING: THIS HAS TO RECEIVE THE SQUEEZED MATRICES. NO BATCH SIZE!
    """
    if attn_matrices[0].shape[0] == 1:
        for attn_matrix in range(len(attn_matrices)):
            attn_matrices[attn_matrix] = attn_matrices[attn_matrix].squeeze()
        
    if attn_matrices[0].ndim != 2:
        raise ValueError("Attention matrices must be a list/array of 2D matrices (n_query, n_key)")
    
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
        attn_with_residual = (1 - alpha) * attn + alpha * np.eye(attn.shape[0])
        
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


def get_token_by_token_attention(model, tokenizer, prompt_text, max_new_tokens=512, target_answer=None):
    """
    Captures the full attention matrices for each generated token across all layers.
    
    This function performs token-by-token generation and captures the attention matrices
    at each step, preserving the full attention state (all heads, all layers) for each
    token generation step.
    Stops after generating the first complete sentence.
    
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
        # tuple (layer_1, layer_2, layer_3, ...)
        # each layer is torch.tensor of shape (batch_size (1), heads, seq_len, seq_len)
        
        # Then generate tokens
        for _ in range(max_new_tokens):
            # Forward pass with attention outputs
            outputs = model(current_ids, output_attentions=True, return_dict=True)
            
            # FIXME: PAVLE what is this?
            # why dont we just use generate fn from transformers
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
            
            # Check if we've completed a sentence
            current_text = tokenizer.decode(current_ids[0][input_length:], skip_special_tokens=True)
            if any(end_of_sentence in current_text for end_of_sentence in ['.', '!', '?', target_answer]):
                break
    
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

def get_cached_attention(attn_dir, model_tuple, prompt_id, prompt_difficulty, prompt_category, prompt_n_shots):
    family, size, variant = model_tuple
    model_identifier = f"{family}_{size}_{variant}"
    
    cached_attention_files = [el for el in os.listdir(attn_dir)
                        if "prompt_attention_values" in el and el.endswith(".npy") 
                        and f"_{prompt_id}" in el and model_identifier in el]
    
    cached_attentions = filter_prompts(cached_attention_files, prompt_difficulty, prompt_category, prompt_n_shots, model_identifier)
    return cached_attentions


def load_attns(models, prompt_path, prompt_id, prompt_difficulty, prompt_category, prompt_n_shots, analysis_type="tokenwise"):
    if analysis_type == "tokenwise":
        return load_attn_tokenwise(
            models,
            prompt_path, prompt_id, prompt_difficulty, prompt_category, prompt_n_shots
            )
    else:
        raise ValueError(f"Invalid analysis type: {analysis_type}")


def load_attn_tokenwise(models, prompt_path, prompt_id, prompt_difficulty, prompt_category, prompt_n_shots, max_new_tokens=50):
    attn_dicts = []
    
    for model_tuple in models:
        prompts = Prompts(prompt_path)
        prompt_object = prompts.get_prompt(
            prompt_id=prompt_id,
            difficulty=prompt_difficulty,
            category=prompt_category,
            n_shots=prompt_n_shots
        )
        prompt = prompt_object['prompt']
        target_answer = prompt_object['target_answer']

        # Get tokenwise attentions
        model_family, model_size, model_variant = model_tuple
        model, tokenizer = get_model_and_tokenizer(model_family, model_size, model_variant)
        attention_data = get_token_by_token_attention(model, tokenizer, prompt, max_new_tokens=max_new_tokens, target_answer=target_answer)

        # Return the full attention data dictionary instead of just attn matrices
        attn_dicts.append(attention_data)
        
    return attn_dicts


def extract_attention_matrices_from_attention_data(attention_data):
    total_attention_matrices = []
    for model_i in range(len(attention_data)):
        model_data = attention_data[model_i]
        attention_matrices = model_data['attention_matrices']
        total_attention_matrices.append(attention_matrices)
    return total_attention_matrices
