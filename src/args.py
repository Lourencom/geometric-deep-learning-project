import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model_size", type=str, default="large", help="Size of the model to use ('small', 'large', or 'huge')")
    
    # Prompt arguments
    parser.add_argument("--prompt_path", type=str, default="data/prompts.json", help="Path to the prompts JSON file")
    parser.add_argument("--prompt_difficulty", type=str, default="medium", help="Difficulty level of the prompt")
    parser.add_argument("--prompt_category", type=str, default=None, help="Category of the prompt")
    parser.add_argument("--prompt_n_shots", type=int, default=None, help="Number of shots for few-shot learning")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="data/attn", help="Directory to save output files")
    parser.add_argument("--attn_dir", type=str, default="data/attn", help="Directory containing attention files")
    
    return parser.parse_args() 