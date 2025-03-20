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


def filter_prompts(prompts, prompt_difficulty, prompt_category, prompt_n_shots, model_size):
    if prompt_difficulty is not None:
        prompts = [p for p in prompts if prompt_difficulty in p]
    if prompt_category is not None:
        prompts = [p for p in prompts if prompt_category in p]
    if prompt_n_shots is not None:
        prompts = [p for p in prompts if prompt_n_shots in p]
    if model_size is not None:
        prompts = [p for p in prompts if model_size in p]
    return prompts


def add_system_instruction(question_prompt):
    system_instruction = "Answer the following question with a concise, one-word or one-number response. Do not include any additional commentary."
    prompt = f"{system_instruction}\n\n{question_prompt}"
    return prompt
