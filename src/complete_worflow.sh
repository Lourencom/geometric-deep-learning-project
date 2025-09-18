# Run the main analysis for all models



# 1. RUN THIS (MUST BE ON SINGLE GPU OR WILL GIVE ERROR, didnt have time to check why)

python src/main.py --aggregate_heads_fn entropy --entropy_alpha 1.0 --prompt_ids 1 2 4 5 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 --output_dir "results/iclr_results/" --models "mistral,8b,instruct" "mistral,24b,instruct" "qwen,1.5b,instruct" "qwen,3b,instruct" "qwen,7b,instruct" "llama,1b,instruct" "llama,8b,instruct" "gemma,2b,instruct" "gemma,9b,instruct" "gemma,27b,instruct"



# 2. Generate plots

python src/feature_distribution_lineplot.py

python src/analyze_model_families.py

python src/plot_feature_distributions.py
