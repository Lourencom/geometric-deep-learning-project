import torch
import os

from utils import get_git_root, get_model_and_tokenizer
from prompts import Prompts


def run_model(args):
    model, tokenizer = get_model_and_tokenizer(args.model_size)

    prompts = Prompts(args.prompt_path)       
    prompt = prompts.get_prompt(
        prompt_id=args.prompt_id,
        difficulty=args.prompt_difficulty,
        category=args.prompt_category,
        n_shots=args.prompt_n_shots
    )

    prompt_text = prompt['prompt']
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        prompt_outputs = model(**inputs, output_attentions=True)

    with torch.no_grad():
        # Get the length of input tokens to separate prompt from generation
        input_length = inputs.input_ids.shape[1]
        generated = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=4,  # Enable beam search
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,  # Use eos_token_id instead of pad_token_id
            early_stopping=True,
            #do_sample=False,  # Deterministic generation
            #temperature=1.0,
            return_dict_in_generate=True,
            no_repeat_ngram_size=3,  # Prevent repetitive text
            output_attentions=True,
        )
        # Only decode the newly generated tokens
        answer = tokenizer.decode(generated[0][input_length:], skip_special_tokens=True)
    
    intermediate_outputs = generated
    
    return [prompt_outputs, intermediate_outputs], answer, prompt_text, prompt['difficulty'], prompt['category'], prompt['n_shots']


def store_answer(args, answer, prompt_text, prompt_difficulty, prompt_category, prompt_n_shots):
    git_root_path = get_git_root()
    output_dir = os.path.join(git_root_path, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the prompt and the answer to a text file for reference
    answer_file = os.path.join(output_dir, f"generated_answer_{args.model_size}_{prompt_difficulty}_{prompt_category}_{prompt_n_shots}.txt")
    with open(answer_file, "w") as f:
        f.write("Prompt:\n")
        f.write(prompt_text + "\n\n")
        f.write("Generated Answer:\n")
        f.write(answer)
    return answer_file
