from args import get_args
from model import run_model, store_answer
from attention import extract_attention
from visualization import plot_features
from prompts import Prompts
from attention import load_attns
from graph_features import GraphFeatures
import os

def analyze_prompt(args, prompt_id):
    """Run the analysis for a single prompt"""
    args.prompt_id = prompt_id
    analysis_type = args.analysis_type
    print(f"\nAnalyzing prompt {prompt_id}...")

    # Load prompt text and metadata
    prompts = Prompts(args.prompt_path)
    prompt_data = prompts.get_prompt(prompt_id=prompt_id)
    prompt_text = prompt_data['prompt']
    
    base_filename = f"prompt_{prompt_id}"
    
    # Load attention data and model answers
    stored_prompt_attns = load_attns(args, models=args.models, attn_dir=args.attn_dir, save=True, tokenwise=analysis_type == "tokenwise")
    
    # Load model answers from saved files
    model_answers = {}
    for model_tuple in args.models:
        family, size, variant = model_tuple
        model_identifier = f"{family}_{size}_{variant}"
        answer_file = os.path.join(args.output_dir, f"generated_answer_{model_identifier}_{prompt_id}.txt")
        
        if os.path.exists(answer_file):
            with open(answer_file, 'r') as f:
                content = f.read()
                # Extract answer part after "Generated Answer:"
                answer = content.split("Generated Answer:\n")[-1].strip()
                model_answers[model_identifier] = answer
        else:
            model_answers[model_identifier] = "Answer file not found"
    
    # Create graph features for each model with different strategies
    graph_strategies = [
        {"mode": "top_k", "top_k": 3},
        {"mode": "top_k", "top_k": 5},
        {"mode": "threshold", "threshold": 0.1},
        {"mode": "threshold", "threshold": 0.2},
    ]
    
    for strategy in graph_strategies:
        graph_features = {}
        strategy_name = f"{strategy['mode']}_{strategy.get('top_k', strategy.get('threshold'))}"
        
        for i, model_tuple in enumerate(args.models):
            family, size, variant = model_tuple
            model_identifier = f"{family}_{size}_{variant}"
            
            # Create graph features with current strategy
            graph_features[model_identifier] = GraphFeatures(
                stored_prompt_attns[i],
                prompt_attn=True,
                remove_attention_sink=True,
                analysis_type=analysis_type,
                plot_raw=True,
                save_path=os.path.join(args.output_dir, f"{base_filename}_{model_identifier}_no_sink_{analysis_type}_{strategy_name}_raw"),
                **strategy
            )

            if args.plot_matrices:
                matrix_filename = os.path.join(
                    args.output_dir, 
                    f"{base_filename}_{model_identifier}_no_sink_{analysis_type}_{strategy_name}"
                )
                # Plot both raw and processed matrices
                graph_features[model_identifier].plot_attention_matrices(
                    save_path=matrix_filename,
                    mode=analysis_type,
                    plot_raw=True,
                )

        # Pass model answers to plot_features
        plot_features(
            graph_features,
            args.models,
            analysis_type,
            prompt_text,
            prompt_data,
            model_answers,
            os.path.join(args.output_dir, f"{base_filename}_features_{strategy_name}")
        )


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.attn_dir, exist_ok=True)

    for prompt_id in args.prompt_ids:
        analyze_prompt(args, prompt_id)

    print("\nDone")
