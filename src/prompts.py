import json
import os
import random
from utils import get_git_root

class Prompts:
    def __init__(self, prompts_path):
        if not prompts_path.startswith('/'):
            prompts_path = os.path.join(get_git_root(), prompts_path)
        with open(prompts_path, 'r') as file:
            self.prompts = json.load(file)
        pass

    def get_prompt(self, difficulty=None, category=None, n_shots=None):
        potential_prompts = self.prompts
        if difficulty is not None:
            potential_prompts = [p for p in potential_prompts if p['difficulty'] == difficulty]
        if category is not None:
            potential_prompts = [p for p in potential_prompts if category in p['category']]
        if n_shots is not None:
            potential_prompts = [p for p in potential_prompts if p['n_shots'] == n_shots]
        
        sampled_prompt = random.choice(potential_prompts)
        return sampled_prompt