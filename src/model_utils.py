"""Utilities for model generation and token sampling"""
import torch

def sample_next_token(current_ids, next_token_logits, temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.0):
    """Sample the next token using various decoding strategies"""
    # Apply temperature
    if temperature > 0:
        next_token_logits = next_token_logits / temperature
    
    # Apply repetition penalty
    if repetition_penalty != 1.0:
        for token_id in current_ids[0]:
            next_token_logits[0, token_id] /= repetition_penalty
    
    # Apply top-k filtering
    if top_k > 0:
        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
        next_token_logits[indices_to_remove] = -float('Inf')
    
    # Apply top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        next_token_logits[0, indices_to_remove] = -float('Inf')
    
    # Sample from the filtered distribution
    probs = torch.softmax(next_token_logits, dim=-1)
    next_token_id = torch.multinomial(probs, num_samples=1)
    return next_token_id
