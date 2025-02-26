import torch
import os
from utils import get_git_root
from prompts import Prompts
from constants import Constants

import transformers
from transformers import AutoTokenizer


def run_model(args):
    # Unpack the current model tuple
    family, size, variant = args.current_model
    model, tokenizer = get_model_and_tokenizer(family, size, variant)

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


def get_model_and_tokenizer(family, size, variant):
    """
    family: str, one of ["llama", "bloom"]
    size: str, one of ["small", "large", "huge"] for llama,
          or one of ["560m", "1b1", "1b7", "3b", "7b1"] for bloom
    variant: str, either "causal" or "instruct"
    """
    if family.lower() == "llama":
        # existing Llama code, e.g.
        if size == "small":
            if variant == "causal":
                model_id = Constants.LLAMA_SMALL_CAUSAL
            elif variant == "instruct":
                model_id = Constants.LLAMA_SMALL_INSTRUCT
            else:
                raise ValueError("Invalid variant for llama")
        elif size == "large":
            if variant == "causal":
                model_id = Constants.LLAMA_LARGE_CAUSAL
            elif variant == "instruct":
                model_id = Constants.LLAMA_LARGE_INSTRUCT
            else:
                raise ValueError("Invalid variant for llama")
        elif size == "huge":
            if variant == "instruct":
                model_id = Constants.LLAMA_HUGE_INSTRUCT
            else:
                raise ValueError("No causal huge variant for llama")
        else:
            raise ValueError("Invalid llama size")
    
    elif family.lower() == "bloom":
        # For BLOOM, use the corresponding string IDs
        if size == "560m":
            if variant == "causal":
                model_id = Constants.BLOOM_560M_CAUSAL
            elif variant == "instruct":
                model_id = Constants.BLOOM_560M_INSTRUCT
            else:
                raise ValueError("Invalid variant for bloom")
        elif size == "1b1":
            if variant == "causal":
                model_id = Constants.BLOOM_1B1_CAUSAL
            elif variant == "instruct":
                model_id = Constants.BLOOM_1B1_INSTRUCT
            else:
                raise ValueError("Invalid variant for bloom")
        elif size == "1b7":
            if variant == "causal":
                model_id = Constants.BLOOM_1B7_CAUSAL
            elif variant == "instruct":
                model_id = Constants.BLOOM_1B7_INSTRUCT
            else:
                raise ValueError("Invalid variant for bloom")
        elif size == "3b":
            if variant == "causal":
                model_id = Constants.BLOOM_3B_CAUSAL
            elif variant == "instruct":
                model_id = Constants.BLOOM_3B_INSTRUCT
            else:
                raise ValueError("Invalid variant for bloom")
        elif size == "7b1":
            if variant == "causal":
                model_id = Constants.BLOOM_7B1_CAUSAL
            elif variant == "instruct":
                model_id = Constants.BLOOM_7B1_INSTRUCT
            else:
                raise ValueError("Invalid variant for bloom")
        else:
            raise ValueError("Invalid bloom size")
    else:
        raise ValueError("Unsupported model family")
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer
