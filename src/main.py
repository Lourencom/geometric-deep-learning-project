import argparse
from model import run_model, store_answer
from attention import extract_attention

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="large")
    parser.add_argument("--prompt_path", type=str, default="data/prompts.json")
    parser.add_argument("--prompt_difficulty", type=str, default="medium")
    parser.add_argument("--prompt_category", type=str, default=None)
    parser.add_argument("--prompt_n_shots", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="data/attn")
    args = parser.parse_args()

    #breakpoint()

    outputs, answer, prompt_text, prompt_difficulty, prompt_category, prompt_n_shots = run_model(args)
    extract_attention(args, outputs)
    store_answer(args, answer, prompt_text, prompt_difficulty, prompt_category, prompt_n_shots)
