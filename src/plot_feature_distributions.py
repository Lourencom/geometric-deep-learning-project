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


models_by_families = {'mistral': ['mistral_8b_instruct', 'mistral_24b_instruct'], 'qwen': ['qwen_1.5b_instruct', 'qwen_3b_instruct', 'qwen_7b_instruct'], 'llama': ['llama_1b_instruct', 'llama_8b_instruct'], 'gemma': ['gemma_2b_instruct', 'gemma_9b_instruct', 'gemma_27b_instruct']}

features_by_families_by_models = {feature_name: {family: {model: [] for model in models_by_families[family]} for family in models_by_families.keys()} for feature_name in feature_names}

fig, axs = plt.subplots(figsize=(20, 20), nrows=3, ncols=1)
for i, feature in enumerate(feature_names):

    # Sample data
    groups = list(models_by_families.keys())
    samples_per_group = [[f"{' '.join(model.split('_')[:-1])}" for model in models_by_families[family]] for family in groups]

    # Flatten sample names and assign random heights
    samples = [s for sublist in samples_per_group for s in sublist]
    heights = [np.median(results[model.replace(" ", "_")+"_instruct"][feature]) for model in samples]

    # Create a DataFrame
    df = pd.DataFrame({"Sample": samples, "Height": heights})
    df["Group"] = np.repeat(groups, [len(g) for g in samples_per_group])

    # Create bar plot
    ax = axs[i]
    sns.barplot(data=df, x="Sample", y="Height", hue="Group", dodge=False, palette="muted", ax=ax)

    # Improve layout
    plt.xticks(rotation=45, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    ax.set_title(feature)

plt.suptitle("Feature distribution for different models and features")
plt.show()
fig.savefig(os.path.join(out_dir, f"feature_distributions.png"), dpi=300)