import json
import matplotlib.pyplot as plt
import numpy as np
import os
import git
import seaborn as sns
import pandas as pd

def get_git_root():
    repo = git.Repo('.', search_parent_directories=True)
    return repo.git.rev_parse("--show-toplevel")


def get_model_family_from_name(model_name):
    return model_name.split('_')[0]

feature_names = [
    'clustering', 
    'average_degree',
    'pagerank'
]

out_dir = os.path.join(get_git_root(), "families")
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(get_git_root(), "entropy/all_comparisons_2/prompt_1/features/features_top_k_20.json"), "r") as f:
    results = json.load(f)


models_by_families = {'mistral': ['8b', '24b'], 'qwen': ['1.5b', '3b', '7b'], 'llama': ['1b', '8b'], 'gemma': ['2b', '9b', '27b']}

features_by_families_by_models = {feature_name: {family: {model: [] for model in models_by_families[family]} for family in models_by_families.keys()} for feature_name in feature_names}


# Populate the data structure
for model_name, features in results.items():
    family = get_model_family_from_name(model_name)
    size = model_name.split('_')[1]
    
    for feature_name in feature_names:
        if feature_name in features:
            features_by_families_by_models[feature_name][family][size].append(features[feature_name])

# Plot each feature
for feature in feature_names:
    # Create a list to store the data for the DataFrame
    data = []
    
    # Extract median values for each model and family
    for family, size_data in features_by_families_by_models[feature].items():
        for size, values in size_data.items():
            if values:  # Only include if we have data
                data.append({
                    'Family': family,
                    'Model Size': size,
                    'Model': f"{family}_{size}",
                    'Value': np.median(values)
                })
    
    # Create DataFrame for seaborn
    df = pd.DataFrame(data)
    
    # Set aesthetic parameters
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    # Create the plot
    # Sort by model size for consistent ordering
    order = sorted(df['Model'].unique(), key=lambda x: (x.split('_')[0], x.split('_')[1]))
    
    # Create the plot with seaborn
    ax = sns.barplot(
        data=df,
        x='Family',
        y='Value',
        hue='Model Size',
        palette='viridis',
        errorbar=None
    )
    
    # Customize the plot
    ax.set_title(f'{feature} by Model Family and Size')
    ax.set_ylabel(feature.capitalize())
    ax.set_xlabel('Model Family')
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8)
    
    # Adjust legend
    plt.legend(title='Model Size', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{feature}_distribution_seaborn.png'))
    plt.close()
