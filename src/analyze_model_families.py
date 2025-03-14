import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from pathlib import Path

def analyze_model_families(prompt_dir):
    # Load the data
    responses_path = prompt_dir / 'features' / 'responses_top_k_20.json'
    features_path = prompt_dir / 'features' / 'features_top_k_20.json'
    
    with open(responses_path, 'r') as f:
        responses = json.load(f)
    with open(features_path, 'r') as f:
        features = json.load(f)

    # Define model families
    model_families = {
        'mistral': ['mistral_8b_instruct', 'mistral_24b_instruct'],
        'qwen': ['qwen_1.5b_instruct', 'qwen_3b_instruct', 'qwen_7b_instruct'],
        'llama': ['llama_1b_instruct', 'llama_8b_instruct'],
        'gemma': ['gemma_2b_instruct', 'gemma_9b_instruct', 'gemma_27b_instruct']
    }

    # Get all metrics from the first model's data
    first_model = list(features.keys())[0]
    metrics = list(features[first_model].keys())
    
    # Calculate number of rows and columns for subplots
    n_metrics = len(metrics)
    n_cols = 4  # We'll use 4 columns
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure and subplots
    fig = plt.figure(figsize=(20, 5*n_rows))
    
    # For each metric, create a subplot
    for idx, metric in enumerate(metrics):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        
        for family_name, family_models in model_families.items():
            # Get data for all models in family
            family_data = []
            max_len = 0
            
            for model in family_models:
                if model in features:
                    data = features[model][metric]
                    family_data.append(data)
                    max_len = max(max_len, len(data))
            
            # Pad shorter sequences with NaN
            padded_data = []
            for data in family_data:
                if len(data) < max_len:
                    padded = data + [np.nan] * (max_len - len(data))
                else:
                    padded = data
                padded_data.append(padded)
            
            # Convert to numpy array for calculations
            family_array = np.array(padded_data)
            
            # Calculate mean and std
            mean = np.nanmean(family_array, axis=0)
            std = np.nanstd(family_array, axis=0)
            
            # Plot mean line with std deviation band
            x = range(len(mean))
            ax.plot(x, mean, label=family_name)
            ax.fill_between(x, mean-std, mean+std, alpha=0.2)

        ax.set_title(metric)
        ax.set_xlabel('Step')
        ax.set_ylabel(metric)
        ax.grid(True)
        
        # Only show legend on first subplot
        if idx == 0:
            ax.legend()

    # Add prompt info as suptitle
    prompt_info = responses['prompt']
    plt.suptitle(f"Prompt {prompt_info['id']}: {prompt_info['text'][:100]}...\n"
                 f"Difficulty: {prompt_info['difficulty']}, "
                 f"Category: {', '.join(prompt_info['category'])}", 
                 wrap=True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for suptitle
    
    # Save plot
    output_path = prompt_dir / 'all_metrics_by_family.png'
    plt.savefig(output_path)
    plt.close()

def main():
    base_dir = Path('../all_comparisons_2')
    
    # Process each prompt directory
    for prompt_dir in base_dir.glob('prompt_*'):
        if prompt_dir.is_dir():
            print(f"Processing {prompt_dir}")
            analyze_model_families(prompt_dir)

if __name__ == "__main__":
    main()
