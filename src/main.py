from args import get_args
from model import run_model, store_answer
from attention import extract_attention
from visualization import plot_features
from prompts import Prompts
from attention import load_attns
from graph_features import GraphFeatures
import os

def analyze_prompt(args, prompt_id, graph_strategies, features):
    """Run the analysis for a single prompt"""
    args.prompt_id = prompt_id
    analysis_type = args.analysis_type
    print(f"\nAnalyzing prompt {prompt_id}...")

    # Create organized directory structure
    prompt_dir = os.path.join(args.output_dir, f"prompt_{prompt_id}")
    answers_dir = os.path.join(prompt_dir, "answers")
    matrices_dir = os.path.join(prompt_dir, "matrices")
    features_dir = os.path.join(prompt_dir, "features")
    
    # Create directories
    os.makedirs(answers_dir, exist_ok=True)
    os.makedirs(matrices_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    
    # Load prompt text and metadata
    prompts = Prompts(args.prompt_path)
    prompt_data = prompts.get_prompt(prompt_id=prompt_id)
    prompt_text = prompt_data['prompt']
        
    # Load attention data and model answers
    stored_prompt_attns = load_attns(args, models=args.models, attn_dir=args.attn_dir, save=True, tokenwise=analysis_type == "tokenwise")
    
    # Load or generate model answers
    model_answers = {}
    for model_tuple in args.models:
        family, size, variant = model_tuple
        model_identifier = f"{family}_{size}_{variant}"
        answer_file = os.path.join(answers_dir, f"{model_identifier}.txt")
        
        if os.path.exists(answer_file):
            with open(answer_file, 'r') as f:
                content = f.read()
                # Extract answer part after "Generated Answer:"
                if "Generated Answer:" in content:
                    answer = content.split("Generated Answer:\n")[-1].strip()
                else:
                    # If the file doesn't contain the expected format, store the whole content
                    answer = content.strip()
                model_answers[model_identifier] = answer
        else:
            # If we don't have a stored answer, we need to generate one
            args.current_model = model_tuple
            _, answer, _, _, _, _ = run_model(args, answer_dir=answers_dir)
            model_answers[model_identifier] = answer

    # Plot raw matrices once per model (not per strategy)
    if args.plot_matrices:
        raw_matrices_dir = os.path.join(matrices_dir, "raw")
        os.makedirs(raw_matrices_dir, exist_ok=True)
        
        for i, model_tuple in enumerate(args.models):
            family, size, variant = model_tuple
            model_identifier = f"{family}_{size}_{variant}"
            
            # Create temporary graph features object for raw matrix plotting
            temp_graph_features = GraphFeatures(
                stored_prompt_attns[i],
                remove_attention_sink=True,
                analysis_type=analysis_type,
                plot_raw=True
            )

            matrix_filename = os.path.join(
                raw_matrices_dir, 
                f"{model_identifier}"
            )
            
            # Plot raw matrices only once
            temp_graph_features.plot_attention_matrices(
                save_path=matrix_filename,
                mode="raw",
            )
    
    # Create graph features for each model with different strategies    
    for strategy in graph_strategies:
        graph_features = {}
        strategy_name = f"{strategy['mode']}_{strategy.get('top_k', strategy.get('threshold'))}"
        
        # Create strategy-specific directories
        strategy_matrices_dir = os.path.join(matrices_dir, strategy_name)
        os.makedirs(strategy_matrices_dir, exist_ok=True)
        
        for i, model_tuple in enumerate(args.models):
            family, size, variant = model_tuple
            model_identifier = f"{family}_{size}_{variant}"
            
            # Create graph features with current strategy
            graph_features[model_identifier] = GraphFeatures(
                stored_prompt_attns[i],
                remove_attention_sink=True,
                analysis_type=analysis_type,
                plot_raw=True,
                save_path=os.path.join(strategy_matrices_dir, f"{model_identifier}_{analysis_type}_raw"),
                **strategy
            )

            if args.plot_matrices:
                matrix_filename = os.path.join(
                    strategy_matrices_dir,
                    f"{model_identifier}_{analysis_type}"
                )

                graph_features[model_identifier].plot_attention_matrices(
                    save_path=matrix_filename,
                    mode=analysis_type,
                )

        # Pass model answers to plot_features
        plot_features(
            features,
            graph_features,
            args.models,
            prompt_text,
            prompt_data,
            model_answers,
            os.path.join(features_dir, f"{strategy_name}")
        )


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.attn_dir, exist_ok=True)

    graph_strategies = [
        {"mode": "top_k", "top_k": 3},
        {"mode": "top_k", "top_k": 5},
        {"mode": "threshold", "threshold": 0.1},
        {"mode": "threshold", "threshold": 0.2},
    ]

    features = [
        'clustering', 
        'average_shortest_path_length', 
        'forman_ricci',
        #'ollivier_ricci', # does not work, parallelism errors
        'average_degree',
        'connectivity',
        'sparseness',
        'hubs',
        'clusters',
        'communities',
        'fourier',
        'commute_time_efficiency',
        'pagerank',
        'eigenvector_centrality',
        'cycle_count',
    ]

    for prompt_id in args.prompt_ids:
        analyze_prompt(args, prompt_id, graph_strategies, features)

    print("\nDone")
