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


data_dir = os.path.join(get_git_root(), "results/iclr_results/")
#additional_dir = os.path.join(get_git_root(), "pavle_asked/all_models/")
prompt_ids = ["prompt_1", "prompt_2", "prompt_3", "prompt_4", "prompt_5", "prompt_8", "prompt_9", "prompt_10", "prompt_11", "prompt_12", "prompt_13", "prompt_14", "prompt_15", "prompt_16", "prompt_17", "prompt_18", "prompt_19", "prompt_20", "prompt_21", "prompt_22"]

# Updated model list including additional models
models_by_families = {
    'mistral': ['mistral_8b_instruct', 'mistral_24b_instruct'], 
    'qwen': ['qwen_1.5b_instruct', 'qwen_3b_instruct', 'qwen_7b_instruct', 'qwen_14b_instruct', 'qwen_32b_instruct'], 
    'llama': ['llama_1b_instruct', 'llama_8b_instruct', 'llama_70b_instruct'], 
    'gemma': ['gemma_2b_instruct', 'gemma_9b_instruct', 'gemma_27b_instruct']
}

# Dictionary to store feature values across all prompts
# Structure: {feature_name: {family: {model: [values_from_different_prompts]}}}
all_prompts_data = {feature_name: {family: {model: [] for model in models} 
                    for family, models in models_by_families.items()} 
                   for feature_name in feature_names}

# Load data from the main directory
for prompt_id in prompt_ids:
    try:
        file_path = os.path.join(data_dir, f"{prompt_id}/features/features_top_k_20.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                results = json.load(f)
                
                # Extract feature values for each model and store them
                for feature_name in feature_names:
                    for family, models in models_by_families.items():
                        for model in models:
                            if model in results and feature_name in results[model]:
                                feature_value = results[model][feature_name]
                                # Handle if feature_value is a list by taking the median
                                if isinstance(feature_value, list) and feature_value:
                                    feature_value = np.median(feature_value)
                                
                                # Only add if it's a valid numeric value
                                if isinstance(feature_value, (int, float)) and not np.isnan(feature_value):
                                    all_prompts_data[feature_name][family][model].append(feature_value)
    except FileNotFoundError:
        print(f"Warning: Could not find file for {prompt_id} in main directory")
        continue

'''# Load data from the additional directory
for prompt_id in prompt_ids:
    try:
        file_path = os.path.join(additional_dir, f"{prompt_id}/features/features_top_k_20.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                results = json.load(f)
                
                # Extract feature values for each model and store them
                for feature_name in feature_names:
                    for family, models in models_by_families.items():
                        for model in models:
                            if model in results and feature_name in results[model]:
                                feature_value = results[model][feature_name]
                                # Handle if feature_value is a list by taking the median
                                if isinstance(feature_value, list) and feature_value:
                                    feature_value = np.median(feature_value)
                                
                                # Only add if it's a valid numeric value
                                if isinstance(feature_value, (int, float)) and not np.isnan(feature_value):
                                    all_prompts_data[feature_name][family][model].append(feature_value)
    except FileNotFoundError:
        print(f"Warning: Could not find file for {prompt_id} in additional directory")
        continue'''

# Create the plots with error bars based on prompt variation
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Create a lineplot for each feature
for i, feature_name in enumerate(feature_names):
    # Prepare data for plotting
    plot_data = []
    
    for family in models_by_families:
        for model in models_by_families[family]:
            model_values = all_prompts_data[feature_name][family][model]
            
            if len(model_values) > 0:  # Only include if we have data points
                model_size = extract_model_size(model)
                
                # Calculate mean and std across prompts
                mean_value = np.mean(model_values)
                std_value = np.std(model_values) if len(model_values) > 1 else 0
                
                plot_data.append({
                    'Family': family,
                    'Model_Size': model_size,
                    'Feature_Value': mean_value,
                    'Error': std_value
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
            errorbar=None,  # Don't use built-in CI
            ax=axs[i]
        )
        
        # Add error bars manually for each family
        for j, family in enumerate(df['Family'].unique()):
            family_data = df[df['Family'] == family]
            if not family_data.empty:
                axs[i].errorbar(
                    x=family_data['Model_Size'],
                    y=family_data['Feature_Value'],
                    yerr=family_data['Error'],
                    fmt='none',  # No additional markers
                    ecolor=f'C{j}',  # Match the color of the line
                    alpha=0.5,
                    capsize=5
                )
        
        # Add labels and title
        axs[i].set_xlabel('Model Size (Billions of Parameters)')
        axs[i].set_ylabel(f'{feature_name.replace("_", " ").title()} Value')
        axs[i].set_title(f'{feature_name.replace("_", " ").title()} evolution with model size')
        
        # Adjust legend and layout
        axs[i].legend(title='Model Family')

plt.tight_layout()
plt.subplots_adjust(top=0.85)  # Make room for suptitle
plt.suptitle('Average feature values across prompts and tokens, with std deviation error bars')
out_path = os.path.join(out_dir, f'feature_distribution_lineplot_avg_prompts.png')
plt.savefig(out_path, dpi=300)
print(f"Plot saved to {out_path}")