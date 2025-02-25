from args import get_args
from model import run_model, store_answer
from attention import extract_attention

if __name__ == "__main__":
    args = get_args()
    outputs, answer, prompt_text, prompt_difficulty, prompt_category, prompt_n_shots = run_model(args)
    extract_attention(args, outputs)
    store_answer(args, answer, prompt_text, prompt_difficulty, prompt_category, prompt_n_shots)
