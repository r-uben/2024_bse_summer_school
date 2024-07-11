import json
from openai import OpenAI
from src.utils import Utils
from tqdm import tqdm
import warnings
import contextlib
import pandas as pd

@contextlib.contextmanager
def suppress_botocore_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="botocore")
        yield

class LLM:
    def __init__(self):
        with suppress_botocore_warnings():
            api_key = self._get_api_key()
        self._client = OpenAI(api_key=api_key)

    def get_response(self, prompt, temperature=0.5, max_tokens=50):
        response = self._client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=temperature
        )
        return response
    
    def generate_samples(self, num_samples, classes):
        generated_data = []
        for cls in classes:
            for _ in range(num_samples):
                prompt = f"Generate a text expressing {cls}."
                generated_text = self.get_response(prompt)
                generated_data.append({"text": generated_text, "label": cls})
        return generated_data

    
    def _get_api_key(self):
        with suppress_botocore_warnings():
            secret = Utils.get_secret("OpenAI-BSE-SS-HW-apikey", "eu-central-1")
        secret_dict = json.loads(secret)
        return secret_dict.get("api_key")
    

