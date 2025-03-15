import os
import matplotlib.pyplot as plt
import torch
import pickle
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,7"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
from model import get_model_and_tokenizer
from attention import get_token_by_token_attention
from prompts import Prompts

import numpy as np
from skimage.util import view_as_blocks

import scipy.linalg

def heuristic(A):
    """
    Compute the heuristic score of a matrix.
    """
    # calculate entropy of the matrix
    a_flat = A.flatten()
    entropy = -np.sum(a_flat * np.log(a_flat + 1e-10))
    return entropy * np.mean(A)
    
def extract_blocks(a, blocksize, keep_as_view=False):
    C, M,N = a.shape
    b0, b1 = blocksize
    if M % b0 != 0 or N % b1 != 0:
        raise ValueError("Matrix dimensions must be divisible by block size")
    if keep_as_view==0:
        return a.reshape(C, M//b0,b0,N//b1,b1).swapaxes(-2,-1).reshape(C, M//b0,N//b1,b0,b1)
    else:
        return a.reshape(C, M//b0,b0,N//b1,b1).swapaxes(1,2)


def head_aggregate_single_token(attention_matrix, blocksize=4):
    """Aggregate attention heads for a single token's attention matrix"""


    n_tokens = attention_matrix[0].shape[-1]

    divisible_part = (n_tokens // blocksize) * blocksize

    aggregated_matrices = []
    for j in range(len(attention_matrix)):
        patchable_part = attention_matrix[j][0, :, :divisible_part, :divisible_part]
        # create patches of size 4x4 using numpy
        patches = extract_blocks(patchable_part, (blocksize, blocksize))
        print(patches.shape)
        
        # Convert to numpy if it's a torch tensor
        if isinstance(patches, torch.Tensor):
            patches = patches.cpu().to(torch.float16).numpy()
        
        # Calculate Laplacian scores for all patches
        num_heads, nx, ny = patches.shape[:3]
        flat_patches = patches.reshape(num_heads * nx * ny, blocksize, blocksize)
        scores = np.array([heuristic(p.astype(np.float32)) for p in flat_patches]).reshape(num_heads, nx, ny)
        
        # Find best head and select patches
        best_heads = np.argmax(scores, axis=0)
        x_idx, y_idx = np.indices((nx, ny))
        aggregated_patches = patches[best_heads, x_idx, y_idx]
        
        # Transform aggregated patches back to full attention matrix
        rows = [np.hstack([aggregated_patches[i, j] for j in range(ny)]) for i in range(nx)]
        reconstructed_attention = np.vstack(rows)

        # make it causal 
        reconstructed_attention = np.tril(reconstructed_attention, k=0)

        final_attn = np.zeros((n_tokens, n_tokens))
        final_attn[:divisible_part, :divisible_part] = reconstructed_attention
        # the rest should just be the mean of the attention matrices per heads
        final_attn[divisible_part:, divisible_part:] = np.mean(attention_matrix[j].cpu().to(torch.float16).numpy()[0,:], axis=0)[divisible_part:, divisible_part:]
        
        aggregated_matrices.append(final_attn)

    return aggregated_matrices

def head_aggregation(attention_matrices, blocksize=4):
    """Wrapper function to aggregate attention heads for all tokens"""
    aggregated_matrices = []
    for i in range(len(attention_matrices)):
        aggregated = head_aggregate_single_token(attention_matrices[i], blocksize)
        aggregated_matrices.append(aggregated)
    return aggregated_matrices
            

if __name__ == "__main__":
    prompts_path = "data/prompts.json"

    prompt_id = 1
    prompt = Prompts(prompts_path).get_prompt(prompt_id)['prompt']


    model, tokenizer = get_model_and_tokenizer(
        "llama",
        "1b",
        "instruct"
    )

    # Generate tokens and collect attention matrices
    save_dir = Path("saved_attention_data")
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir / f"prompt_{prompt_id}_attention.pt"
    pickle_path = save_dir / f"prompt_{prompt_id}_attention.pkl"

    # Check if attention data already exists
    if save_path.exists():
        print(f"Loading existing attention data from {save_path}")
        outputs = torch.load(save_path)
    else:
        # Generate new attention data
        outputs = get_token_by_token_attention(model, tokenizer, prompt, max_new_tokens=200)
        
        # Save the outputs to a file for later visualization
        torch.save(outputs, save_path)
        print(f"Saved attention data to {save_path}")
        
        # Also save as pickle for compatibility
        with open(pickle_path, 'wb') as f:
            pickle.dump(outputs, f)
        print(f"Saved attention data to {pickle_path}")

    # Create a directory for the visualizations
    output_dir = "attention_visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    attention_matrices = outputs['attention_matrices']

    head_aggregation(attention_matrices)
