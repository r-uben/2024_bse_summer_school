import json
from openai import OpenAI
from src.llm import LLM
from src.utils import Utils
from tqdm import tqdm
import warnings
import contextlib
import pandas as pd
import numpy as np

@contextlib.contextmanager
def suppress_botocore_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="botocore")
        yield

class ZeroShotClassifier(LLM):
    def __init__(self, 
                 data, 
                 prompt_function, 
                 num_percentiles=20, 
                 num_samples=32,
                 temperature=0):
        super().__init__()  # Initialize the parent class (LLM)
        self.__data = data
        self.__num_percentiles = num_percentiles
        self.__results = None
        self.__prompt_function = prompt_function
        self.__num_samples = num_samples
        self.__temperature = temperature

    def _get_api_key(self):
        with suppress_botocore_warnings():
            secret = Utils.get_secret("OpenAI-BSE-SS-HW-apikey", "eu-central-1")
        secret_dict = json.loads(secret)
        return secret_dict.get("api_key")

    @property
    def percentages(self):
        return [i / self.__num_percentiles for i in range(1, 21, 4)]

    @property
    def results(self):
        if self.__results is None:
            self.__results = self.get_results()
        return self.__results

    def get_results(self):
        results = {}
        #for percentage in tqdm(self.percentages, desc="Processing percentages"):
        percentage = 0.2
        split_data = Utils.get_split_sample(self.__data.train, percentage)
        train_sample = split_data['train']
        test_sample = split_data['valid']
        
        # Create DataFrames from the train and test samples
        train_df = pd.DataFrame(train_sample)
        test_df = pd.DataFrame(test_sample)
        
        # Get few-shot examples from the training sample
        few_shot_examples = train_df.sample(n=min(self.__num_samples, len(train_df))) if len(train_df) > 0 else None
        percentage = len(few_shot_examples) / len(train_df)
        predictions = self.classify(test_df, few_shot_examples, temperature=self.__temperature)
        results[percentage] = Utils.metrics(test_df['label'], predictions)
        
        return Utils.convert_results_to_dataframe(results)

    def classify(self, data, few_shot_examples=None, temperature=0):
        results = []
        for _, item in tqdm(data.iterrows(), total=len(data), desc="Classifying"):
            prompt = self.__prompt_function(item, few_shot_examples)
            response = self.get_response(prompt, temperature = temperature) 
            predicted_label = response.choices[0].message.content.strip().lower()
            try:
                results.append(int(predicted_label))
            except ValueError:
                print(f"Invalid predicted label: {predicted_label}")
                results.append(np.nan)
        return results