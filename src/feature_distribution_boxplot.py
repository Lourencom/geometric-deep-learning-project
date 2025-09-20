import json
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import get_git_root

def get_model_family_from_name(model_name):
    return model_name.split('_')[0]

feature_names = [
    'clustering', 
    'average_degree',
    'pagerank'
]

out_dir = os.path.join(get_git_root(), "families")
os.makedirs(out_dir, exist_ok=True)

# Read feature files from all prompts
prompts = ['prompt_1', 'prompt_2', 'prompt_4', 'prompt_5', 'prompt_8', 'prompt_9', 'prompt_10', 'prompt_11', 'prompt_12', 'prompt_13', 'prompt_14', 'prompt_15', 'prompt_16', 'prompt_17', 'prompt_18', 'prompt_19', 'prompt_20', 'prompt_21', 'prompt_22']
all_results = []
for prompt in prompts:
    feature_file = os.path.join(get_git_root(), f"results/iclr_results/{prompt}/features/features_top_k_20.json")
    with open(feature_file, "r") as f:
        all_results.append(json.load(f))

# Combine results from all prompts
combined_results = {}
for model_name in all_results[0].keys():
    combined_results[model_name] = {feature: [] for feature in feature_names}
    for results in all_results:
        for feature_name in feature_names:
            combined_results[model_name][feature_name].extend(results[model_name][feature_name])

families = {'mistral', 'qwen', 'llama', 'gemma'}
features_by_families = {feature_name: {family: [] for family in families} for feature_name in feature_names}

for model_name in combined_results.keys():
    for feature_name in feature_names:
        model_family = get_model_family_from_name(model_name)
        features_by_families[feature_name][model_family].extend(combined_results[model_name][feature_name])

fig, axs = plt.subplots(1, len(feature_names), figsize=(20, 10))
for i, feature_name in enumerate(feature_names):
    data = [features_by_families[feature_name][family] for family in families]
    axs[i].boxplot(data, patch_artist=True, labels=families)
    axs[i].set_title(feature_name)

plt.suptitle("Boxplot feature distributions for different model families and features (combined across prompts)")
plt.tight_layout()
plt.show()
fig.savefig(os.path.join(out_dir, "feature_boxplots_families_entropy.png"), dpi=300)
