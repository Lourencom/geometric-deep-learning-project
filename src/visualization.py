import seaborn as sns
import matplotlib.pyplot as plt

def plot_attention_matrices(attention_matrices, aggregation="head"):
    assert aggregation in ["head", "layer"], "Must aggregate across 'head's or 'layer's"
    
    if aggregation == "head":
    # Visualize each layer's attention matrix (averaged across heads) using Seaborn
        for layer_idx, attn_tensor in enumerate(attention_matrices):
            # Remove the batch dimension (assuming batch_size = 1)
            attn = attn_tensor[0]  # Shape: (num_heads, seq_len, seq_len)
            
            # Average across heads to get a single matrix per layer
            attn_avg = attn.mean(axis=0)  # Shape: (seq_len, seq_len)
            plt.figure(figsize=(8, 6))
            sns.heatmap(attn_avg, cmap="Reds")
            plt.title(f"Layer {layer_idx} (averaged over heads)")
            plt.xlabel("Key Tokens")
            plt.ylabel("Query Tokens")
            plt.show()
    else:
        raise ValueError(f"Not yet implemented")