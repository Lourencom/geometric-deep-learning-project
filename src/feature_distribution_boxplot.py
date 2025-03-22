
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import git


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


families = {'mistral', 'qwen', 'llama', 'gemma'}
features_by_families = {feature_name: {family: [] for family in families} for feature_name in feature_names}
for model_name in results.keys():
    for feature_name in feature_names:
        model_family = get_model_family_from_name(model_name)
        
        features_by_families[feature_name][model_family].append(results[model_name][feature_name])


for feature_name in feature_names:
    for family in families:
        features_by_families[feature_name][family] = [el for sublist in features_by_families[feature_name][family] for el in sublist] # flatten the list

fig, axs = plt.subplots(1, len(feature_names), figsize=(20, 10))
for i, feature_name in enumerate(feature_names):
    
    axs[i].boxplot(features_by_families[feature_name].values(), patch_artist=True, labels=families)
    axs[i].set_title(feature_name)

plt.suptitle("Boxplot feature distributions for different model families and features")
plt.show()
fig.savefig(os.path.join(out_dir, "families_feature_distributions.png"), dpi=300)
