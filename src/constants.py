import transformers
import torch
from transformers import AutoTokenizer

class Constants:
    # LLAMA causal models (from 1B to 8B)
    LLAMA_1B_CAUSAL = "meta-llama/Llama-3.2-1B"
    LLAMA_8B_CAUSAL = "meta-llama/Meta-Llama-3-8B"

    # LLAMA instruct models (from 1B to 70B)
    LLAMA_1B_INSTRUCT = "meta-llama/Llama-3.2-1B-Instruct"
    LLAMA_8B_INSTRUCT = "meta-llama/Meta-Llama-3-8B-Instruct"
    LLAMA_70B_INSTRUCT = "meta-llama/Llama-3.3-70B-Instruct"
    
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

    # GEMMA Models
    GEMMA_2B_INSTRUCT = "google/gemma-2-2b-it"
    GEMMA_9B_INSTRUCT = "google/gemma-2-9b-it"
    GEMMA_27B_INSTRUCT = "google/gemma-2-27b-it"

    # QWEN Models
    QWEN_25_05B_INSTRUCT = "Qwen/Qwen2.5-0.5B-Instruct"
    QWEN_25_15B_INSTRUCT = "Qwen/Qwen2.5-1.5B-Instruct"
    QWEN_25_3B_INSTRUCT = "Qwen/Qwen2.5-3B-Instruct"
    QWEN_25_7B_INSTRUCT = "Qwen/Qwen2.5-7B-Instruct"
    QWEN_25_14B_INSTRUCT = "Qwen/Qwen2.5-14B-Instruct"
    QWEN_25_32B_INSTRUCT = "Qwen/Qwen2.5-32B-Instruct"
    QWEN_25_72B_INSTRUCT = "Qwen/Qwen2.5-72B-Instruct"


    # MISTRAL Models
    MISTRAL_8B_INSTRUCT = "mistralai/Ministral-8B-Instruct-2410"
    MISTRAL_24B_INSTRUCT = "mistralai/Mistral-Small-24B-Instruct-2501"


model_mapping = {
    "llama": {
        "1b": {"causal": Constants.LLAMA_1B_CAUSAL, "instruct": Constants.LLAMA_1B_INSTRUCT},
        "8b": {"causal": Constants.LLAMA_8B_CAUSAL, "instruct": Constants.LLAMA_8B_INSTRUCT},
        "70b": {"instruct": Constants.LLAMA_70B_INSTRUCT},
    },
    "bloom": {
        "560m": {"causal": Constants.BLOOM_560M_CAUSAL, "instruct": Constants.BLOOM_560M_INSTRUCT},
        "1b1": {"causal": Constants.BLOOM_1B1_CAUSAL, "instruct": Constants.BLOOM_1B1_INSTRUCT},
        "1b7": {"causal": Constants.BLOOM_1B7_CAUSAL, "instruct": Constants.BLOOM_1B7_INSTRUCT},
        "3b": {"causal": Constants.BLOOM_3B_CAUSAL, "instruct": Constants.BLOOM_3B_INSTRUCT},
        "7b1": {"causal": Constants.BLOOM_7B1_CAUSAL, "instruct": Constants.BLOOM_7B1_INSTRUCT},
    },
    "gemma": {
        "2b": {"instruct": Constants.GEMMA_2B_INSTRUCT},
        "9b": {"instruct": Constants.GEMMA_9B_INSTRUCT},
        "27b": {"instruct": Constants.GEMMA_27B_INSTRUCT},
    },
    "qwen": {
        "0.5b": {"instruct": Constants.QWEN_25_05B_INSTRUCT},
        "1.5b": {"instruct": Constants.QWEN_25_15B_INSTRUCT},
        "3b": {"instruct": Constants.QWEN_25_3B_INSTRUCT},
        "7b": {"instruct": Constants.QWEN_25_7B_INSTRUCT},
        "14b": {"instruct": Constants.QWEN_25_14B_INSTRUCT},
        "32b": {"instruct": Constants.QWEN_25_32B_INSTRUCT},
        "72b": {"instruct": Constants.QWEN_25_72B_INSTRUCT},
    },
    "mistral": {
        "8b": {"instruct": Constants.MISTRAL_8B_INSTRUCT},
        "24b": {"instruct": Constants.MISTRAL_24B_INSTRUCT},
    }
}


def get_model_and_tokenizer(family, size, variant):
    """
    family: str, one of ["llama", "bloom", "gemma", "qwen"]
    size: str, eg. ["1b", "8b", "70b"] for llama,
          or ["560m", "1b1", "1b7", "3b", "7b1"] for bloom
    variant: str, either "causal" or "instruct"
    """
    family = family.lower()
    size = size.lower()
    variant = variant.lower()

    if family not in model_mapping:
        raise ValueError(f"Invalid model family: {family}. Options are {list(model_mapping.keys())}")
    if size not in model_mapping[family]:
        raise ValueError(f"Invalid model size: {size} for family: {family}. Options are {list(model_mapping[family].keys())}")
    if variant not in model_mapping[family][size]:
        raise ValueError(f"Invalid model variant: {variant} for family: {family}, size: {size}. Options are {list(model_mapping[family][size].keys())}")

    model_id = model_mapping[family][size][variant]
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer
