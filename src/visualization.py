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


def plot_features(features, graph_features, models, prompt_text, prompt_data, model_answers, save_path):
    """Plot features with enhanced information."""
    
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // 2
    
    # Adjust figure size to be more compact
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
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
        
        # Add vertical line at prompt boundary using passed prompt_length
        #ax.axvline(x=prompt_length, color='red', linestyle='--', alpha=0.5, label='Prompt End')
        # PROMPT ENDS AT 1 BASICALLY
        
        ax.set_xlabel("Tokens")
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
        f"Prompt {prompt_data['id']} | "
        f"Difficulty: {prompt_data['difficulty']} | "
        f"Category: {prompt_data['category']} | "
        f"N-shots: {prompt_data['n_shots']}\n"
        f"Prompt: {prompt_text[:100]}...\n"
    )
    
    # Add model answers more compactly
    answer_info = "Answers: "
    for model_tuple in models:
        family, size, variant = model_tuple
        model_identifier = f"{family}_{size}_{variant}"
        answer = model_answers.get(model_identifier, "No answer available")
        if len(answer) > 100:
            answer = answer[:100] + "..."
        answer_info += f"\n{model_identifier}: {answer}"
    
    full_info = prompt_info + answer_info
    
    # Adjust title position and spacing
    fig.suptitle(full_info, y=1.0, fontsize=9, wrap=True)
    
    # Tighter layout with less spacing
    plt.tight_layout()
    # Adjust spacing after tight_layout to accommodate the title
    plt.subplots_adjust(top=0.9)
    
    plt.savefig(save_path + ".png", bbox_inches='tight')
    plt.close(fig)
