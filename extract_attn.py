import torch
import transformers
from transformers import AutoTokenizer

from constants import Constants

model_choice = 'large'

if model_choice == 'small':
    model = transformers.AutoModelForCausalLM.from_pretrained(Constants.SMALL_MODEL_NAME_CAUSAL_LM)
    tokenizer = AutoTokenizer.from_pretrained(Constants.SMALL_MODEL_NAME_CAUSAL_LM)
elif model_choice == 'large':
    model = transformers.AutoModelForCausalLM.from_pretrained(Constants.LARGE_MODEL_NAME_CAUSAL_LM)
    tokenizer = AutoTokenizer.from_pretrained(Constants.LARGE_MODEL_NAME_CAUSAL_LM)
else:
    raise ValueError(f"Invalid model choice: {model_choice}")



prompt = """
TASK QUERY

Input:
86
64
Output:
868686
646464
686868
464646
868686
646464

Input:
79
43
Output:
797979
434343
979797
343434
797979
434343

Input:
32
78
Output:
"""

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=10)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
