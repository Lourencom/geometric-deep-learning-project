import torch
import os
import numpy as np
from prompts import Prompts
from constants import get_model_and_tokenizer
from attention import get_token_by_token_attention, get_cached_attention, extract_attention
from utils import store_answer, save_attention_data

def run_model(
        prompt_path, prompt_id, prompt_difficulty, prompt_category, prompt_n_shots,
        current_model, answer_dir
        ):
    # Unpack the current model tuple
    family, size, variant = current_model
    model, tokenizer = get_model_and_tokenizer(family, size, variant)

    prompts = Prompts(prompt_path)       
    prompt = prompts.get_prompt(
        prompt_id=prompt_id,
        difficulty=prompt_difficulty,
        category=prompt_category,
        n_shots=prompt_n_shots
    )

    prompt_text = prompt['prompt']
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        prompt_outputs = model(**inputs, output_attentions=True)

    with torch.no_grad():
        # Get the length of input tokens to separate prompt from generation
        input_length = inputs.input_ids.shape[1]
        
        # Use generate with streaming to get attention for each token
        generated = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=1,  # Disable beam search for token-by-token attention (beam search complicates attention tracking)
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,  # Get scores for each token
            output_attentions=True,  # Get attention matrices
            output_hidden_states=True,  # Get hidden states

            # deterministic genertion and control parameters to encourage shorter responses
            do_sample=False,
            min_length=1,
            repetition_penalty=1.2,
            length_penalty=1.0,
            forced_eos_token_id=tokenizer.eos_token_id,
        )
        
        # Extract attention matrices from generation
        attention_matrices = []
        if hasattr(generated, 'attentions') and generated.attentions is not None:
            # For models that return attentions directly
            attention_matrices = generated.attentions
        elif hasattr(generated, 'decoder_attentions') and generated.decoder_attentions is not None:
            # For encoder-decoder models
            attention_matrices = generated.decoder_attentions
            
        # Process attention matrices for each generated token
        token_attentions = attention_matrices # process_token_attentions(attention_matrices)
        
        # Only decode the newly generated tokens
        answer = tokenizer.decode(generated.sequences[0][input_length:], skip_special_tokens=True)
    
    # Store both the raw generated output and processed attention matrices
    intermediate_outputs = {
        'generated': generated,
        'token_attentions': token_attentions
    }

    store_answer(current_model, answer_dir, answer, prompt_text, prompt_id)
    
    return [prompt_outputs, intermediate_outputs], answer, prompt_text, prompt['difficulty'], prompt['category'], prompt['n_shots']


def load_attn_layerwise(args, attn_path, models_to_process, save=False):
    attn_dicts = []
    
    for model_tuple in models_to_process:
        cached_attentions = get_cached_attention(attn_path, model_tuple, args.prompt_id, args.prompt_difficulty, args.prompt_category, args.prompt_n_shots)
        
        if len(cached_attentions) == 0:
            outputs, *_ = run_model(
                args.prompt_path, args.prompt_id, args.prompt_difficulty, args.prompt_category, args.prompt_n_shots,
                model_tuple, args.output_dir,
            )
            attn_dict = extract_attention(model_tuple, args.prompt_id, outputs, attn_path, save=save)["prompt_attns"]
        else:
            attn_dict = np.load(os.path.join(attn_path, cached_attentions[0]), allow_pickle=True)
        attn_dicts.append(attn_dict)
    return attn_dicts


def process_token_attentions(attention_matrices):
    """
    Process the attention matrices to extract per-token attention patterns.
    
    Args:
        attention_matrices: The attention matrices from the model generation
        
    Returns:
        A list of processed attention data for each generated token
    """
    # This structure will depend on the exact format of attention matrices
    # from your specific model, but here's a general approach
    token_attentions = []
    
    if not attention_matrices:
        return token_attentions
        
    # The structure of attention matrices varies by model architecture
    # This is a general approach that may need adjustment for your specific model
    try:
        # For most models using generate with output_attentions=True
        # attention_matrices is typically a tuple of tuples
        for step_idx, step_attentions in enumerate(attention_matrices):
            # Each step corresponds to generating one new token
            # step_attentions contains attention for all layers for this step
            
            # Aggregate attention across layers (you can modify this strategy)
            # For example, you might want to look at specific layers or heads
            layer_attentions = []
            for layer_idx, layer_attention in enumerate(step_attentions):
                # layer_attention shape: [batch_size, num_heads, seq_len, seq_len]
                # We typically have batch_size=1 in generation
                
                # Average across attention heads (or select specific heads)
                # Shape becomes [seq_len, seq_len]
                avg_attention = layer_attention.squeeze().mean(dim=0)
                layer_attentions.append(avg_attention)
            
            # Store attention for this token generation step
            token_attentions.append({
                'token_idx': step_idx,
                'layer_attentions': layer_attentions,
                # You can add more metrics here, like entropy, max attention, etc.
            })
    except Exception as e:
        # Fallback for different attention matrix structures
        print(f"Error processing attention matrices: {e}")
        # You might need to implement different processing logic
        # depending on your model's output structure
    
    return token_attentions


def batch_analyze_prompts(model, tokenizer, prompts, output_dir=None, max_new_tokens=100):
    """
    Analyzes multiple prompts and collects their attention patterns.
    
    Args:
        model: The HuggingFace model to use
        tokenizer: The tokenizer corresponding to the model
        prompts: List of prompt texts to analyze
        output_dir: Directory to save the results (None to skip saving)
        max_new_tokens: Maximum number of tokens to generate for each prompt
        
    Returns:
        A dictionary mapping prompt indices to attention data
    """
    results = {}
    
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}")
        
        # Get token-by-token attention
        attention_data = get_token_by_token_attention(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        
        # Store the results
        results[i] = {
            'prompt': prompt,
            'attention_data': attention_data
        }
        
        # Save to disk if output directory is provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"prompt_{i}_attention.pt")
            save_attention_data(attention_data, output_path)
            results[i]['saved_path'] = output_path
    
    return results
