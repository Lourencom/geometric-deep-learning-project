#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import git
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

def get_git_root():
    repo = git.Repo('.', search_parent_directories=True)
    return repo.git.rev_parse("--show-toplevel")

def get_model_family_from_name(model_name):
    return model_name.split('_')[0]

def extract_model_size(model_name):
    # Extract model size in billions (e.g., 8b from mistral_8b_instruct)
    import re
    match = re.search(r'(\d+\.?\d*)b', model_name)
    if match:
        return float(match.group(1))
    return 0  # Default case if no match found

# Define the correctness data from statistical_significance_test.py
# Map from model name used in features to model name used in results_by_prompt
model_name_mapping = {
    'llama_1b_instruct': 'Llama 1b',
    'llama_8b_instruct': 'Llama 8b',
    'gemma_2b_instruct': 'Gemma 2b',
    'gemma_9b_instruct': 'Gemma 9b',
    'gemma_27b_instruct': 'Gemma 27b',
    'qwen_1.5b_instruct': 'Qwen 1.5b',
    'qwen_3b_instruct': 'Qwen 3b',
    'qwen_7b_instruct': 'Qwen 7b',
    'mistral_8b_instruct': 'Mistral 8b',
    'mistral_24b_instruct': 'Mistral 24b'
}

# Correctness data from statistical_significance_test.py
results_by_prompt = {
    'prompt_1': {
        'Llama 1b': False,
        'Llama 8b': True,
        'Gemma 2b': False,
        'Gemma 9b': True,
        'Gemma 27b': True,
        'Qwen 1.5b': False,
        'Qwen 3b': False,
        'Qwen 7b': True,
        'Mistral 8b': True,
        'Mistral 24b': True
    },
    'prompt_2': {
        'Llama 1b': False,
        'Llama 8b': True,
        'Gemma 2b': True,
        'Gemma 9b': True,
        'Gemma 27b': True,
        'Qwen 1.5b': True,
        'Qwen 3b': True,
        'Qwen 7b': True,
        'Mistral 8b': True,
        'Mistral 24b': True
    },
    'prompt_3': {
        'Llama 1b': False,
        'Llama 8b': True,
        'Gemma 2b': False,
        'Gemma 9b': True,
        'Gemma 27b': True,
        'Qwen 1.5b': True,
        'Qwen 3b': True,
        'Qwen 7b': True,
        'Mistral 8b': True,
        'Mistral 24b': True
    }
}

def main():
    # Set the feature names
    feature_names = ['clustering', 'average_degree', 'pagerank']
    
    # Set the data directory and prompt IDs
    data_dir = os.path.join(get_git_root(), "results/iclr_results/")
    prompt_ids = ["prompt_1", "prompt_2", "prompt_4", "prompt_5", "prompt_8", "prompt_9", "prompt_10", "prompt_11", "prompt_12", "prompt_13", "prompt_14", "prompt_15", "prompt_16", "prompt_17", "prompt_18", "prompt_19", "prompt_20", "prompt_21", "prompt_22"] # Include prompt_4 if it exists
    
    # Dictionary to store feature values for each feature, grouped by correctness across all families
    # Structure: {feature_name: {'correct': [...], 'incorrect': [...]}}
    all_features_data = {feature: {'correct': [], 'incorrect': []} for feature in feature_names}
    
    # Dictionary to store per-family feature values (for individual analysis)
    # Structure: {family: {feature_name: {'correct': [...], 'incorrect': [...]}}}
    family_feature_data = {}
    
    # Collect all data points for analysis
    all_data_points = []
    
    # Process data from all prompts
    for prompt_id in prompt_ids:
        try:
            prompt_num = int(prompt_id.split('_')[1])
            
            # Load feature data
            with open(os.path.join(data_dir, f"{prompt_id}/features/features_top_k_20.json"), "r") as f:
                features_data = json.load(f)
                
                # Extract feature values and organize by correctness
                for model_name, features in features_data.items():
                    family = get_model_family_from_name(model_name)
                    
                    # Initialize family data if not already present
                    if family not in family_feature_data:
                        family_feature_data[family] = {
                            feature: {'correct': [], 'incorrect': []} 
                            for feature in feature_names
                        }
                    
                    # Map model name to the format used in results_by_prompt
                    mapped_model_name = model_name_mapping.get(model_name)
                    
                    # Skip if we don't have correctness data for this model
                    if mapped_model_name is None or prompt_num > 3:
                        continue
                        
                    # Get correctness for this model on this prompt
                    result_prompt_key = f'prompt_{prompt_num}'
                    is_correct = results_by_prompt.get(result_prompt_key, {}).get(mapped_model_name, None)
                    
                    # Skip if we don't have correctness data for this prompt
                    if is_correct is None:
                        continue
                    
                    # Group by correctness
                    correctness_group = 'correct' if is_correct else 'incorrect'
                    
                    # Add feature values to the appropriate groups
                    for feature_name in feature_names:
                        if feature_name in features:
                            feature_value = features[feature_name]
                            
                            # Handle if feature_value is a list by taking the median
                            if isinstance(feature_value, list) and feature_value:
                                feature_value = np.median(feature_value)
                            
                            # Only add if it's a valid numeric value
                            if isinstance(feature_value, (int, float)) and not np.isnan(feature_value):
                                # Add to both family-specific and global data collections
                                family_feature_data[family][feature_name][correctness_group].append(feature_value)
                                all_features_data[feature_name][correctness_group].append(feature_value)
                                
                                # Add as a data point for the combined dataframe
                                all_data_points.append({
                                    'Family': family,
                                    'Feature': feature_name,
                                    'Value': feature_value,
                                    'Correctness': correctness_group,
                                    'Prompt': prompt_id,
                                    'Model': mapped_model_name
                                })
                    
        except FileNotFoundError:
            print(f"Warning: Could not find file for {prompt_id}")
            continue
    
    # Convert all data points to DataFrame for easier analysis
    df = pd.DataFrame(all_data_points)
    if not df.empty:
        print(f"Collected {len(df)} data points across all models and prompts")
        
        # Create a directory for the outputs
        out_dir = os.path.join(get_git_root(), "feature_correctness")
        os.makedirs(out_dir, exist_ok=True)
        
        # Save the full dataset for reference
        df.to_csv(os.path.join(out_dir, 'all_feature_data.csv'), index=False)
    else:
        print("No data collected. Check path and data availability.")
        return
    
    # Create figure for per-feature plots (aggregated across all families)
    plt.figure(figsize=(15, 5))
    
    # Track Mann-Whitney test results
    results_data = []
    
    # Analyze each feature with aggregated data from all families
    for i, feature_name in enumerate(feature_names):
        correct_values = np.array(all_features_data[feature_name]['correct'])
        incorrect_values = np.array(all_features_data[feature_name]['incorrect'])
        
        plt.subplot(1, 3, i+1)
        
        print(f"\nAnalyzing feature: {feature_name} (all families combined)")
        print(f"  Correct samples: {len(correct_values)}")
        print(f"  Incorrect samples: {len(incorrect_values)}")
        
        # Only perform test if we have sufficient data
        if len(correct_values) >= 3 and len(incorrect_values) >= 3:
            # Perform Mann-Whitney U test
            stat, p_value = mannwhitneyu(correct_values, incorrect_values, alternative='two-sided')
            
            # Interpretation
            alpha = 0.05
            significance = "Significant" if p_value < alpha else "Not significant"
            
            print(f"  U statistic: {stat:.2f}, p-value: {p_value:.4f}")
            print(f"  {significance} difference between correct and incorrect responses")
            
            # Create boxplot
            boxplot_data = pd.DataFrame({
                'Feature Value': np.concatenate([correct_values, incorrect_values]),
                'Correctness': ['Correct'] * len(correct_values) + ['Incorrect'] * len(incorrect_values)
            })
            
            sns.boxplot(x='Correctness', y='Feature Value', data=boxplot_data)
            plt.title(f"{feature_name.capitalize()}\np={p_value:.4f}")
            
            # Add individual data points
            sns.stripplot(x='Correctness', y='Feature Value', data=boxplot_data,
                        size=4, color='black', alpha=0.3)
            
            # Add result to our collection
            results_data.append({
                'Family': 'All',
                'Feature': feature_name,
                'U_statistic': stat,
                'p_value': p_value,
                'Significant': p_value < alpha,
                'Correct_mean': np.mean(correct_values),
                'Incorrect_mean': np.mean(incorrect_values),
                'Sample_size_correct': len(correct_values),
                'Sample_size_incorrect': len(incorrect_values)
            })
        else:
            print("  Insufficient data for statistical test")
            plt.title(f"{feature_name.capitalize()}\nInsufficient data")
    
    plt.tight_layout()
    plt.suptitle('Difference in Feature distributions based on LLM answer correctness (averaged over all prompts and model families)', y=1.05)
    
    # Save the combined plot
    plt.savefig(os.path.join(out_dir, 'combined_feature_correctness.png'), dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to {os.path.join(out_dir, 'combined_feature_correctness.png')}")
    
    # Create per-family plots as well (if we have enough data)
    if family_feature_data:
        num_families = len(family_feature_data)
        fig, axs = plt.subplots(num_families, len(feature_names), figsize=(15, 4*num_families))
        
        # Make axs a 2D array to handle single family case
        if num_families == 1:
            axs = np.expand_dims(axs, axis=0)
        
        # Process each family and feature
        for i, (family, features) in enumerate(family_feature_data.items()):
            print(f"\nAnalyzing family: {family}")
            
            for j, feature_name in enumerate(feature_names):
                correct_values = np.array(features[feature_name]['correct'])
                incorrect_values = np.array(features[feature_name]['incorrect'])
                
                print(f"  {feature_name}:")
                print(f"    Correct samples: {len(correct_values)}")
                print(f"    Incorrect samples: {len(incorrect_values)}")
                
                # Only perform test if we have sufficient data
                if len(correct_values) >= 3 and len(incorrect_values) >= 3:
                    # Perform Mann-Whitney U test
                    stat, p_value = mannwhitneyu(correct_values, incorrect_values, alternative='two-sided')
                    
                    # Interpretation
                    alpha = 0.05
                    significance = "Significant" if p_value < alpha else "Not significant"
                    
                    print(f"    U statistic: {stat:.2f}, p-value: {p_value:.4f}")
                    print(f"    {significance} difference between correct and incorrect responses")
                    
                    results_data.append({
                        'Family': family,
                        'Feature': feature_name,
                        'U_statistic': stat,
                        'p_value': p_value,
                        'Significant': p_value < alpha,
                        'Correct_mean': np.mean(correct_values),
                        'Incorrect_mean': np.mean(incorrect_values),
                        'Sample_size_correct': len(correct_values),
                        'Sample_size_incorrect': len(incorrect_values)
                    })
                    
                    # Create boxplot
                    boxplot_data = pd.DataFrame({
                        'Feature Value': np.concatenate([correct_values, incorrect_values]),
                        'Correctness': ['Correct'] * len(correct_values) + ['Incorrect'] * len(incorrect_values)
                    })
                    
                    sns.boxplot(x='Correctness', y='Feature Value', data=boxplot_data, ax=axs[i, j])
                    axs[i, j].set_title(f"{family} - {feature_name}\np={p_value:.4f}")
                    
                    # Add dots for individual data points
                    sns.stripplot(x='Correctness', y='Feature Value', data=boxplot_data,
                                size=4, color='black', alpha=0.3, ax=axs[i, j])
                    
                else:
                    print("    Insufficient data for statistical test")
                    axs[i, j].set_title(f"{family} - {feature_name}\nInsufficient data")
        
        # Adjust layout and save the plot
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.suptitle('Feature Values by Correctness of Response (Per Family)')
        
        # Save plot
        out_path = os.path.join(out_dir, 'per_family_feature_correctness.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Per-family plot saved to {out_path}")
    
    # Save results as CSV
    if results_data:
        results_df = pd.DataFrame(results_data)
        csv_path = os.path.join(out_dir, 'mann_whitney_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    
    return results_data

if __name__ == "__main__":
    main() 