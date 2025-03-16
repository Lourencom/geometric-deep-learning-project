"""Utilities for model generation and token sampling"""
import torch

def sample_next_token(current_ids, next_token_logits, temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.0):
    """Sample the next token using various decoding strategies"""
    # Check for NaN values and replace them
    if torch.isnan(next_token_logits).any():
        # Replace NaN values with a very negative number (effectively removing them from consideration)
        next_token_logits = torch.nan_to_num(next_token_logits, nan=-float('inf'))
    
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
        next_token_logits[indices_to_remove] = -float('inf')
    
    # Apply top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        next_token_logits[0, indices_to_remove] = -float('inf')
    
    # Sample from the filtered distribution
    # Handle potential NaN/Inf values in the logits before softmax
    next_token_logits = torch.nan_to_num(next_token_logits, nan=-float('inf'), posinf=-float('inf'), neginf=-float('inf'))
    
    # Ensure we have valid logits before softmax
    max_logit = next_token_logits.max()
    if max_logit == -float('inf'):
        # If all logits are -inf, set a uniform distribution
        next_token_logits = torch.zeros_like(next_token_logits)
    
    probs = torch.softmax(next_token_logits, dim=-1)
    
    # Ensure no NaN values in probabilities
    probs = torch.nan_to_num(probs, nan=1e-10)
    
    # Ensure probabilities sum to 1


    probs = probs / probs.sum()
    
    next_token_id = torch.multinomial(probs, num_samples=1)
    return next_token_id
