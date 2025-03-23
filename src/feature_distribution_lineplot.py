import json
import matplotlib.pyplot as plt
import numpy as np
import os
import git
import seaborn as sns
import pandas as pd
import re

def get_git_root():
    repo = git.Repo('.', search_parent_directories=True)
    return repo.git.rev_parse("--show-toplevel")


def get_model_family_from_name(model_name):
    return model_name.split('_')[0]


def extract_model_size(model_name):
    # Extract model size in billions (e.g., 8b from mistral_8b_instruct)
    match = re.search(r'(\d+\.?\d*)b', model_name)
    if match:
        return float(match.group(1))
    return 0  # Default case if no match found


feature_names = [
    'clustering', 
    'average_degree',
    'pagerank'
]

out_dir = os.path.join(get_git_root(), "families")
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(get_git_root(), "entropy/all_comparisons_2/prompt_1/features/features_top_k_20.json"), "r") as f:
    results = json.load(f)


models_by_families = {'mistral': ['mistral_8b_instruct', 'mistral_24b_instruct'], 'qwen': ['qwen_1.5b_instruct', 'qwen_3b_instruct', 'qwen_7b_instruct'], 'llama': ['llama_1b_instruct', 'llama_8b_instruct'], 'gemma': ['gemma_2b_instruct', 'gemma_9b_instruct', 'gemma_27b_instruct']}

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
# Create a lineplot for each feature
for i, feature_name in enumerate(feature_names):
    
    
    # Prepare data for plotting
    plot_data = []
    
    for family in models_by_families:
        for model in models_by_families[family]:
            if model in results and feature_name in results[model]:
                model_size = extract_model_size(model)
                feature_value = results[model][feature_name]
                
                # Handle if feature_value is a list by taking the average
                if isinstance(feature_value, list):
                    if feature_value:  # Check if list is not empty
                        feature_value = np.median(feature_value)
                    else:
                        continue  # Skip if empty list
                
                plot_data.append({
                    'Family': family,
                    'Model_Size': model_size,
                    'Feature_Value': feature_value
                })
    
    # Convert to DataFrame for easy plotting with seaborn
    if plot_data:
        df = pd.DataFrame(plot_data)
        
        # Create lineplot
        sns.set_style("whitegrid")
        sns.lineplot(
            data=df, 
            x='Model_Size', 
            y='Feature_Value', 
            hue='Family',
            marker='o', 
            markersize=8,
            linewidth=2,
            ax=axs[i]
        )
        
        # Add labels and title
        axs[i].set_xlabel('Model Size (Billions of Parameters)')
        axs[i].set_ylabel(f'{feature_name.replace("_", " ").title()} Value')
        axs[i].set_title(f'{feature_name.replace("_", " ").title()} evolution with model size')
        
        # Adjust legend and layout
        axs[i].legend(title='Model Family')
        #plt.tight_layout()


print("All lineplots generated successfully.")

plt.suptitle('Median feature values evolving with model size, across different model families')
out_path = os.path.join(out_dir, f'feature_distribution_lineplot.png')
plt.savefig(out_path, dpi=300)