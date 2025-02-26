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
        
        # Create a mapping of IDs to prompts for faster lookup
        self.id_to_prompt = {p['id']: p for p in self.prompts}

    def get_prompt(self, prompt_id=None, difficulty=None, category=None, n_shots=None):
        """
        Get a prompt either by ID or by filtering criteria.
        If prompt_id is provided, it takes precedence over other filters.
        """
        if prompt_id is not None:
            if prompt_id not in self.id_to_prompt:
                raise ValueError(f"No prompt found with ID: {prompt_id}")
            return self.id_to_prompt[prompt_id]
        
        # If no ID provided, filter by other criteria
        potential_prompts = self.prompts
        if difficulty is not None:
            potential_prompts = [p for p in potential_prompts if p['difficulty'] == difficulty]
        if category is not None:
            potential_prompts = [p for p in potential_prompts if category in p['category']]
        if n_shots is not None:
            potential_prompts = [p for p in potential_prompts if p['n_shots'] == n_shots]
        
        if not potential_prompts:
            raise ValueError(f"No prompts found matching criteria: difficulty={difficulty}, category={category}, n_shots={n_shots}")
        
        sampled_prompt = random.choice(potential_prompts)
        return sampled_prompt