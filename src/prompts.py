import json
import random

class Prompts:
    def __init__(self, prompts_path):
        with open(prompts_path, 'r') as file:
            self.prompts = json.load(file)

    def get_prompt(self, prompt_name, difficulty=None, category=None, n_shots=None):
        potential_prompts = self.prompts
        if difficulty is not None:
            potential_prompts = [p for p in potential_prompts if p['difficulty'] == difficulty]
        if category is not None:
            potential_prompts = [p for p in potential_prompts if category in p['category']]
        if n_shots is not None:
            potential_prompts = [p for p in potential_prompts if p['n_shots'] == n_shots]
        return random.choice(potential_prompts)['prompt']