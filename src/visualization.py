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
        plt.savefig(savepath)
    plt.close()
