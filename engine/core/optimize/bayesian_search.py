'''
Bayesian search with https://docs.ray.io/en

By the time makes pure random permutations until it finds a solution.
'''
from core.chains.prompt_chain import run_variant
from utils.create_logger import get_logger
from ray import tune
import requests

from difflib import SequenceMatcher
import random


class PromptOptimizer:
    def __init__(
            self, 
            task: str, 
            base_prompt: str, 
            input_example: str, 
            expected_output: str, 
            n_trials: int
        ):
        self.task = task
        self.base_prompt = base_prompt
        self.input_example = input_example
        self.expected_output = expected_output
        self.n_trials = n_trials

        self.best_prompt = base_prompt
        self.best_score = -float("inf")
        self.trials = n_trials

    def call_model(self, prompt: str):
        result = run_variant(
            template=prompt,
            input_text=self.input_example,
            task=self.task
        )

        return result

    def score(self, output: str, expected: str):
        return SequenceMatcher(None, output, expected).ratio()

    def generate_candidate(self, prompt: str):
        mutations = [
            lambda p: p + "\nBe concise.",
            lambda p: p + "\nBe precise.",
            lambda p: p + "\nReturn structured output.",
            lambda p: p.replace("Explain", "Clearly explain"),
        ]

        return random.choice(mutations)(prompt)
    
    def run(self):
        for _ in range(self.n_trials):
            candidate = self.generate_candidate(self.best_prompt)

            result = self.call_model(candidate)

            score = self.score(result.text, self.expected_output)

            if score > self.best_score:
                self.best_score = score
                self.best_prompt = candidate

        return self