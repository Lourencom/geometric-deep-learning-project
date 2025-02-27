import argparse
from utils import relative_to_absolute_path


def parse_model_tuple(s):
    parts = s.split(",")
    if len(parts) != 3:
        raise ValueError(f"Invalid model tuple: {s}, must be in the format 'family,size,variant'")
    family, size, variant = parts
    return family, size, variant


def get_args():
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument(
        "--models",
        type=parse_model_tuple,
        nargs="+",
        default=[("bloom", "560m", "instruct"), ("bloom", "1b7", "instruct")],
        help=(
            "List of models to analyze, provided as tuples in the format 'family,size,variant'. "
            "For example: 'llama,small,causal' 'bloom,1b7,instruct'"
        )
    )
    
    # Prompt arguments
    parser.add_argument("--prompt_path", type=str, default="data/prompts.json", help="Path to the prompts JSON file")
    parser.add_argument("--prompt_ids", type=int, nargs="+", required=True, help="List of prompt IDs to analyze")
    parser.add_argument("--prompt_difficulty", type=str, default=None, help="Difficulty level of the prompt")
    parser.add_argument("--prompt_category", type=str, default=None, help="Category of the prompt")
    parser.add_argument("--prompt_n_shots", type=int, default=None, help="Number of shots for few-shot learning")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="media/feature_plots", help="Directory to save output files")
    parser.add_argument("--attn_dir", type=str, default="data/attn", help="Directory containing attention files")

    # Tokenwise arguments
    parser.add_argument("--analysis_type", type=str, default="tokenwise", help="Whether to use tokenwise attention")

    # Visualization arguments
    parser.add_argument("--plot_matrices", type=bool, default=False, help="Whether to plot attention matrices")
    
    args = parser.parse_args()

    args.output_dir = relative_to_absolute_path(args.output_dir)
    args.attn_dir = relative_to_absolute_path(args.attn_dir)

    return args
