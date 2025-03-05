import transformers
import torch
from transformers import AutoTokenizer

class Constants:
    # LLAMA causal models (from 1B to 8B)
    LLAMA_SMALL_CAUSAL = "meta-llama/Llama-3.2-1B"
    LLAMA_LARGE_CAUSAL = "meta-llama/Meta-Llama-3-8B"

    # LLAMA instruct models (from 1B to 70B)
    LLAMA_SMALL_INSTRUCT = "meta-llama/Llama-3.2-1B-Instruct"
    LLAMA_LARGE_INSTRUCT = "meta-llama/Meta-Llama-3-8B-Instruct"
    LLAMA_HUGE_INSTRUCT = "meta-llama/Llama-3.3-70B-Instruct"
    
    # BLOOM causal models (from 560M to 7B)
    BLOOM_560M_CAUSAL = "bigscience/bloom-560m"
    BLOOM_1B1_CAUSAL = "bigscience/bloom-1b1"
    BLOOM_1B7_CAUSAL = "bigscience/bloom-1b7"
    BLOOM_3B_CAUSAL   = "bigscience/bloom-3b"
    BLOOM_7B1_CAUSAL  = "bigscience/bloom-7b1"
    
    # BLOOM instruct variants (BLOOMZ) corresponding to the above sizes:
    BLOOM_560M_INSTRUCT = "bigscience/bloomz-560m"
    BLOOM_1B1_INSTRUCT = "bigscience/bloomz-1b1"
    BLOOM_1B7_INSTRUCT = "bigscience/bloomz-1b7"
    BLOOM_3B_INSTRUCT   = "bigscience/bloomz-3b"
    BLOOM_7B1_INSTRUCT  = "bigscience/bloomz-7b1"


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

