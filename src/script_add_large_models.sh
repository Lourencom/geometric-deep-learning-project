#!/bin/bash

python main.py --aggregate_heads_fn average --prompt_ids 1 2 4 --output_dir "pavle_asked/all_models" --models "qwen,14b,instruct" "qwen,32b,instruct" "llama,70b,instruct"

echo "All done!"
