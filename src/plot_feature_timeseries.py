import matplotlib.pyplot as plt
import numpy as np
import json

# Load the data from JSON file
file_path = "entropy/all_comparisons_2/prompt_1/features/features_top_k_20.json"
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract clustering feature for model


feature_to_model_map = {
    "average_degree": "qwen_3b_instruct",
    "clustering": "gemma_2b_instruct",
    "pagerank": "gemma_2b_instruct",
}

feature_color_map = {
    "average_degree": "#000080",
    "clustering": "#000080",
    "pagerank": "#b2fba5",
}

for feature_name in feature_to_model_map.keys():
    model_name = feature_to_model_map[feature_name]
    model_data = data[model_name]
    feature_values = model_data[feature_name]

    first_n_steps = 27

    # Create a time axis (just indices for now)
    time_steps = np.arange(len(feature_values))

    # Create the plot
    plt.figure(figsize=(10, 3))
    plt.plot(time_steps[:first_n_steps], feature_values[:first_n_steps], marker='o', linestyle='-', linewidth=2, markersize=6, color=feature_color_map[feature_name])

    # Add semi-transparent fill under the curve
    plt.fill_between(time_steps[:first_n_steps], feature_values[:first_n_steps], alpha=0.3, color=feature_color_map[feature_name])
    if feature_name == "average_degree":
        plt.ylim(18.8, 19.5)

    if feature_name == "clustering":
        plt.ylim(0.0045, 0.009)

    if feature_name == "pagerank":
        plt.ylim(0.42, 0.45)


    # Add title and labels
    plt.title(f'{feature_name.replace("_", " ")}', fontsize=16)
    #plt.xlabel('Time Steps', fontsize=14)
    #plt.ylabel(feature_name, fontsize=14)
    plt.axis('off')
    # Add grid for better readability
    #plt.grid(True, linestyle='--', alpha=0.7)

    # Improve aesthetics
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'results/feature_timeseries/{feature_name}_over_time.png', dpi=300)

    # Also display it
    plt.show() 