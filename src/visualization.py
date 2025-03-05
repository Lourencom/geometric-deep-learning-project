import seaborn as sns
import matplotlib.pyplot as plt

def plot_attention_matrix(attention_matrix, savepath=None):
    # Visualize each layer's attention matrix (averaged across heads) using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention_matrix, cmap="Reds")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.show()
    if savepath:
        plt.savefig(savepath + ".png")
    plt.close()


def plot_features(features, graph_features, models, analysis_type, prompt_text, prompt_data, model_answers, save_path):
    """Plot features with enhanced information."""
    
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    linestyles = ['-', '--', ':', '-.']
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p']
    
    for idx, feature in enumerate(features):
        print(f"Running {feature}...")
        ax = axes[idx]
        
        for i, model_tuple in enumerate(models):
            family, size, variant = model_tuple
            model_identifier = f"{family}_{size}_{variant}"
            
            linestyle = linestyles[i % len(linestyles)]
            marker = markers[i % len(markers)]
            
            feature_values = graph_features[model_identifier].extract(feature)
            ax.plot(feature_values, 
                   label=model_identifier,
                   linestyle=linestyle,
                   marker=marker,
                   markevery=5,
                   markersize=4)
        
        ax.set_xlabel("Layers" if analysis_type == "layerwise" else "Tokens")
        ax.set_ylabel(feature)
        ax.set_title(feature)
        ax.grid(True)
    
    # Remove unused subplots
    for idx in range(len(features), len(axes)):
        fig.delaxes(axes[idx])
    
    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5))
    
    # Add prompt and answer information
    prompt_info = (
        f"Prompt {prompt_data['id']}\n"
        f"Difficulty: {prompt_data['difficulty']}\n"
        f"Category: {prompt_data['category']}\n"
        f"N-shots: {prompt_data['n_shots']}\n\n"
        f"Prompt: {prompt_text[:200]}...\n\n"
    )
    
    # Add model answers
    answer_info = "Model Answers:\n"
    for model_tuple in models:
        family, size, variant = model_tuple
        model_identifier = f"{family}_{size}_{variant}"
        answer = model_answers.get(model_identifier, "No answer available")
        # Truncate long answers
        if len(answer) > 200:
            answer = answer[:200] + "..."
        answer_info += f"\n{model_identifier}:\n{answer}\n"
    
    # Combine prompt and answer information
    full_info = prompt_info + answer_info
    
    # Add the information to the plot
    fig.suptitle(full_info, y=1.02, fontsize=10, wrap=True)
    
    # Adjust figure size to accommodate the text
    plt.subplots_adjust(top=0.85)  # Adjust this value to fit your text
    
    plt.tight_layout()
    plt.savefig(save_path + ".png", bbox_inches='tight')
    plt.close(fig)
