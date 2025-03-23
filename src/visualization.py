import seaborn as sns
import matplotlib.pyplot as plt
from validate_answer import evaluate_answer


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


def plot_responses(prompt_text, prompt_data, model_answers, save_path):
    """Plot model responses in a separate figure."""
    fig = plt.figure(figsize=(12, 6))
    
    # Add prompt and answer information
    prompt_info = (
        f"Prompt {prompt_data['id']} | "
        f"Difficulty: {prompt_data['difficulty']} | "
        f"Category: {prompt_data['category']} | "
        f"N-shots: {prompt_data['n_shots']}\n"
        f"Prompt: {prompt_text}\n"
    )
    
    # Add model answers with validation
    answer_info = "Answers: "
    for model_identifier, answer in model_answers.items():
        is_correct = evaluate_answer(answer, prompt_data['target_answer'])
        status = "✓" if is_correct else "✗"
        if len(answer) > 100:
            answer = answer[:100] + "..."
        # Escape dollar signs to prevent matplotlib from interpreting them as math mode
        answer = answer.replace('$', '\\$')
        answer_info += f"\n{model_identifier} [{status}]: {answer}"
    
    full_info = prompt_info + answer_info
    
    # Add text to figure
    plt.text(0.5, 0.5, full_info, 
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes,
             fontsize=10,
             wrap=True)
    
    # Remove axes
    plt.axis('off')
    
    # Save figure
    plt.savefig(save_path + "_responses.png", bbox_inches='tight')
    plt.close(fig)


def plot_features(features, graph_features, models, prompt_text, prompt_data, model_answers, save_path):
    """Plot features with enhanced information."""
    
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + 1) // n_cols
    
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
        
        ax.set_xlabel("Tokens")
        ax.set_ylabel(feature)
        ax.set_title(feature)
        #ax.grid(True)
    
    # Remove unused subplots
    for idx in range(len(features), len(axes)):
        fig.delaxes(axes[idx])
    
    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5))
    
    # Add prompt as title
    #fig.suptitle("Feature values", y=1.0, fontsize=9, wrap=True)
    
    # Tighter layout with less spacing
    plt.tight_layout()
    # Adjust spacing after tight_layout to accommodate the title
    plt.subplots_adjust(top=0.9)
    
    plt.savefig(save_path + "_features.png", bbox_inches='tight')
    plt.close(fig)
