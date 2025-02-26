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
