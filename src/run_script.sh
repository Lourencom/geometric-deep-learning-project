#!/bin/bash

# Define the base output directory variable
BASE_OUTPUT_DIR="results/"

# echo "First run: Qwen..."
# python main.py --prompt_ids 1 2 4 --output_dir "${BASE_OUTPUT_DIR}/qwen_comparisons_2/" --models "qwen,1.5b,instruct" "qwen,3b,instruct" "qwen,7b,instruct"

#echo "Second run: Gemma..."
#python main.py --prompt_ids 1 2 4 --output_dir "${BASE_OUTPUT_DIR}/gemma_comparisons_2/" --models "gemma,2b,instruct" "gemma,9b,instruct" "gemma,27b,instruct"

#echo "Third run: LLama..."
#python main.py --prompt_ids 1 2 4 --output_dir "${BASE_OUTPUT_DIR}/llama_comparisons_2/" --models "llama,1b,instruct" "llama,8b,instruct"

#echo "Fourth run: Mistral..."
#python main.py --prompt_ids 1 2 4 --output_dir "${BASE_OUTPUT_DIR}/mistral_comparisons_2/" --models "mistral,8b,instruct" "mistral,24b,instruct"

#echo "Compare all models..."
#python main.py --aggregate_heads_fn average --prompt_ids 1 2 4 --output_dir "${BASE_OUTPUT_DIR}/average_compare_all_models/" --models "mistral,8b,instruct" "mistral,24b,instruct" "qwen,1.5b,instruct" "qwen,3b,instruct" "qwen,7b,instruct" "llama,1b,instruct" "llama,8b,instruct" "gemma,2b,instruct" "gemma,9b,instruct" "gemma,27b,instruct"


echo "Compare all models with entropy..."
python main.py --aggregate_heads_fn entropy --entropy_alpha 1.0 --prompt_ids 1 2 4 --output_dir "${BASE_OUTPUT_DIR}/entropy_compare_all_models/" --models "mistral,8b,instruct" "mistral,24b,instruct" "qwen,1.5b,instruct" "qwen,3b,instruct" "qwen,7b,instruct" "llama,1b,instruct" "llama,8b,instruct" "gemma,2b,instruct" "gemma,9b,instruct" "gemma,27b,instruct"



echo "All done!"
