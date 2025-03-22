import os
import numpy as np
import matplotlib.pyplot as plt
from head_agg import compute_row_entropy
from attention import load_attns
import torch
import git

def get_git_root():
    repo = git.Repo('.', search_parent_directories=True)
    return repo.git.rev_parse("--show-toplevel")

def layerwise_entropy(layer_attn, alpha=0.5):
    num_heads, seq_len, _ = layer_attn.shape

    aggregated_layer_matrix = np.zeros((seq_len, seq_len))
    # For each row position
    for row_idx in range(seq_len):
        # Get all attention weights for this row across all heads
        row_attentions = []

        for head_idx in range(num_heads):
            layer_row_attn = layer_attn[head_idx, row_idx, :]
            row_attentions.append(layer_row_attn)
                
        # Compute entropy for each row
        entropies = np.array([compute_row_entropy(row) for row in row_attentions])
        
        # Select head based on alpha parameter
        sorted_indices = np.argsort(entropies)
        # will choose median entropy if alpha = 0.5, highest entropy if alpha = 1.0, lowest entropy if alpha = 0.0
        selected_idx = sorted_indices[int(alpha * (len(sorted_indices) - 1))]
        
        # Use the selected attention pattern
        aggregated_layer_matrix[row_idx, :] = row_attentions[selected_idx]
    
    return aggregated_layer_matrix


def high_frequency_energy(matrix, ax=None, title="", plot_fig=True):  
    # Compute 2D FFT
    fft_matrix = np.fft.fft2(matrix)
    fft_shifted = np.fft.fftshift(fft_matrix)
    
    # Compute magnitude spectrum
    magnitude_spectrum = np.abs(fft_shifted)

    # plot the magnitude spectrum if requested
    if plot_fig and ax is not None:
        ax.imshow(magnitude_spectrum)
        ax.set_title(title)
        ax.set_axis_off()

    # Compute high-frequency energy
    H, W = magnitude_spectrum.shape
    center_x, center_y = H // 2, W // 2
    radius = min(H, W) // 4  # Define cutoff for high frequencies
    high_freq_energy = np.sum(magnitude_spectrum) - np.sum(magnitude_spectrum[center_x-radius:center_x+radius, center_y-radius:center_y+radius])
    
    return high_freq_energy


def analyze_attention_patterns(attns_by_heads, plot_fig=True):
    n_heads = attns_by_heads.shape[0]

    # Create plots only if plot_fig is True
    if plot_fig:
        fig, ax = plt.subplots(n_heads // 8, 8, figsize=(30, 30))
        fig_magnitude_spectrum, ax_magnitude_spectrum = plt.subplots(n_heads // 8, 8, figsize=(30, 30))
    
    per_head_energies = []
    avg_energy = 0
    agg_entropy_energy = 0
    # Analyze each head
    for head_i in range(n_heads):
        attn = attns_by_heads[head_i, :, :]
        
        # Plot the attention matrix if requested
        if plot_fig:
            ax[head_i // 8, head_i % 8].imshow(attn)
            ax[head_i // 8, head_i % 8].set_title(f'Head {head_i}')
            ax[head_i // 8, head_i % 8].set_axis_off()
            
            # Calculate and plot frequency spectrum
            energy = high_frequency_energy(attn, 
                                          ax_magnitude_spectrum[head_i // 8, head_i % 8], 
                                          f'Head {head_i}', 
                                          plot_fig)
        else:
            energy = high_frequency_energy(attn, plot_fig=False)
            
        per_head_energies.append(energy)
    # Save figures if requested
    if plot_fig:
        fig.savefig(os.path.join(out_dir, f'attn_head_{head_i}.png'), dpi=500)
        fig_magnitude_spectrum.savefig(os.path.join(out_dir, f'magnitude_spectrum_head_{head_i}.png'), dpi=500)

    # Compute mean aggregation
    agg_heads = np.mean(attns_by_heads, axis=0)
    
    if plot_fig:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(agg_heads)
        ax.set_title('Aggregated Attention')
        ax.set_axis_off()
        plt.show()

        fig_avg_spectrum, ax_avg_spectrum = plt.subplots(figsize=(10, 10))
        avg_energy = high_frequency_energy(agg_heads, ax_avg_spectrum, 'Averaged head', plot_fig)
    else:
        avg_energy = high_frequency_energy(agg_heads, plot_fig=False)
        


    if plot_fig:
        ax_avg_spectrum.set_title('Aggregated Magnitude Spectrum')
        ax_avg_spectrum.set_axis_off()
        plt.show()
        fig_avg_spectrum.savefig(os.path.join(out_dir, f'magnitude_spectrum_avg_head.png'), dpi=500)

    # Compute entropy aggregation
    agg_heads_entropy = layerwise_entropy(attns_by_heads)

    if plot_fig:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(agg_heads_entropy)
        ax.set_title('Aggregated Attention Entropy')
        ax.set_axis_off()
        plt.show()
        fig.savefig(os.path.join(out_dir, f'entropy_avg_head.png'), dpi=500)

        fig_entropy, ax_entropy = plt.subplots(figsize=(10, 10))
        agg_entropy_energy = high_frequency_energy(agg_heads_entropy, ax_entropy, 'Aggregated heads by entropy', plot_fig)
    else:
        agg_entropy_energy = high_frequency_energy(agg_heads_entropy, plot_fig=False)
        
    
    if plot_fig:
        ax_entropy.set_title('Aggregated Attention Entropy')
        ax_entropy.set_axis_off()
        plt.show()
        fig_entropy.savefig(os.path.join(out_dir, f'magnitude_spectrum_entropy_avg_head.png'), dpi=500)

    return per_head_energies, avg_energy, agg_entropy_energy


if __name__ == "__main__":
    # Load attention data
    attention_data = load_attns(
        [("llama", "8b", "instruct")],
        "data/prompts.json", 1, None,None,None,
        analysis_type="tokenwise",
        )[0]['attention_matrices']
    
    out_dir = os.path.join(get_git_root(), "results/high_freq_visuals")

    plot_fig = False
    all_per_head_energies = []
    all_avg_energies = []
    all_agg_entropy_energies = []
    for t in range(len(attention_data)):
        for l in range(len(attention_data[t])):
            attns_by_heads = attention_data[t][l].cpu().to(torch.float32).numpy()[0]
            per_head_energies, avg_energy, agg_entropy_energy = analyze_attention_patterns(attns_by_heads, plot_fig)
            all_per_head_energies.append(per_head_energies)
            all_avg_energies.append(avg_energy)
            all_agg_entropy_energies.append(agg_entropy_energy)

    mean_per_head_energies = np.mean(all_per_head_energies, axis=0)
    std_per_head_energies = np.std(all_per_head_energies, axis=0)

    mean_avg_energy = np.mean(all_avg_energies)
    std_avg_energy = np.std(all_avg_energies)

    mean_agg_entropy_energy = np.mean(all_agg_entropy_energies)
    std_agg_entropy_energy = np.std(all_agg_entropy_energies)
    
    print(f"Average per head energy: {mean_per_head_energies}")
    print(f"Std per head energy: {std_per_head_energies}")
    print(f"Average average energy: {mean_avg_energy}")
    print(f"Std average energy: {std_avg_energy}")
    print(f"Average aggregated entropy energy: {mean_agg_entropy_energy}")
    print(f"Std aggregated entropy energy: {std_agg_entropy_energy}")

    # Create a single plot with all three distributions
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for boxplot
    data_to_plot = [
        np.array(all_per_head_energies).flatten(),  # Flatten because it's a nested list
        all_avg_energies,
        all_agg_entropy_energies
    ]
    
    # Create boxplot
    box = ax.boxplot(data_to_plot, patch_artist=True, labels=['Individual heads', 'Mean aggregation', 'Entropy-based aggregation (alpha = 0.5)'])
    
    # Set colors for boxes
    colors = ['lightblue', 'lightgreen', 'salmon']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add title and labels
    ax.set_title('Comparison of High Frequency Energy Distributions')
    ax.set_ylabel('Energy')
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.savefig(os.path.join(out_dir, 'energy_distributions_comparison.png'), dpi=300)
    plt.show()