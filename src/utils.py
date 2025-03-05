import subprocess
import os
import torch

def get_git_root():
    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
        ).strip().decode("utf-8")
        return git_root
    except subprocess.CalledProcessError:
        return None  # Not inside a Git repository
    
def relative_to_absolute_path(path):
    return os.path.join(get_git_root(), path)


def store_answer(current_model, output_dir, answer, prompt_text, prompt_id):
    output_dir = relative_to_absolute_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    family, size, variant = current_model
    model_identifier = f"{family}_{size}_{variant}"
    
    # Save the prompt and the answer to a text file for reference
    answer_file = os.path.join(output_dir, f"generated_answer_{model_identifier}_{prompt_id}.txt")
    with open(answer_file, "w") as f:
        f.write("Prompt:\n")
        f.write(prompt_text + "\n\n")
        f.write("Generated Answer:\n")
        f.write(answer)
    return answer_file

def store_attention_data(args, token_attentions, prompt_id):
    """
    Store the attention data for later analysis.
    
    Args:
        args: The command line arguments
        token_attentions: The processed attention data for each token
        prompt_id: The ID of the prompt used for generation
    
    Returns:
        The path to the saved attention data file
    """
    git_root_path = get_git_root()
    output_dir = os.path.join(git_root_path, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the attention data to a PyTorch file
    attention_file = os.path.join(output_dir, f"attention_data_{args.current_model}_{prompt_id}.pt")
    torch.save(token_attentions, attention_file)
    
    return attention_file

def save_attention_data(attention_data, output_path):
    """
    Saves the attention data to disk for later analysis.
    
    Args:
        attention_data: The output from get_token_by_token_attention
        output_path: Path to save the data
        
    Returns:
        The path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert attention matrices to CPU before saving
    cpu_attention_data = attention_data.copy()
    
    # Process attention matrices
    cpu_attention_matrices = []
    for step_attentions in attention_data['attention_matrices']:
        cpu_step_attentions = []
        for layer_attention in step_attentions:
            # Move to CPU
            cpu_layer_attention = layer_attention.cpu()
            cpu_step_attentions.append(cpu_layer_attention)
        cpu_attention_matrices.append(cpu_step_attentions)
    
    cpu_attention_data['attention_matrices'] = cpu_attention_matrices
    
    # Move prompt tokens to CPU
    cpu_attention_data['prompt_tokens'] = attention_data['prompt_tokens'].cpu()
    
    # Save to disk
    torch.save(cpu_attention_data, output_path)
    
    return output_path


def load_attention_data(input_path, device=None):
    """
    Loads the attention data from disk.
    
    Args:
        input_path: Path to the saved attention data
        device: Device to load the tensors to (None for CPU)
        
    Returns:
        The loaded attention data
    """
    # Load from disk
    attention_data = torch.load(input_path)
    
    # Move to specified device if needed
    if device is not None:
        # Process attention matrices
        device_attention_matrices = []
        for step_attentions in attention_data['attention_matrices']:
            device_step_attentions = []
            for layer_attention in step_attentions:
                # Move to device
                device_layer_attention = layer_attention.to(device)
                device_step_attentions.append(device_layer_attention)
            device_attention_matrices.append(device_step_attentions)
        
        attention_data['attention_matrices'] = device_attention_matrices
        
        # Move prompt tokens to device
        attention_data['prompt_tokens'] = attention_data['prompt_tokens'].to(device)
    
    return attention_data
