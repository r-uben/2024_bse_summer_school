from src.llm import LLM
from src.utils import Utils

import json
import pandas as pd  # Import pandas to handle DataFrame operations
import os
from datasets import Dataset, DatasetDict  # Import datasets library
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting data

class LLMData:
    def __init__(self, num_samples_per_class, llm=LLM()):
        self.llm = llm
        self.num_samples_per_class = num_samples_per_class
        self.emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        self.emotion_labels = {
            "sadness": 0,
            "joy": 1,
            "love": 2,
            "anger": 3,
            "fear": 4,
            "surprise": 5
        }
        self.__ds = None
        self.__train = None
        self.__test = None

    def generate_samples(self):
        generated_data = []

        for emotion in self.emotions:
            for _ in range(self.num_samples_per_class):
                prompt = {
                    "instruction": "Generate a text expressing the given emotion.",
                    "emotion": emotion,
                    "notice": "Do not include any numbers in the text."
                }
                prompt = json.dumps(prompt)
                response = self.llm.get_response(prompt)  
                generated_text = response.choices[0].message.content.strip().lower()  # Extract the generated text
                generated_text = generated_text.replace('\n', ' ').replace('\r', '')  # Clean the generated text
                generated_data.append({"text": generated_text, "label": self.emotion_labels[emotion]})

        return generated_data

    def convert_to_dataset_dict(self, generated_data):
        df = pd.DataFrame(generated_data)
        df.reset_index(drop=True, inplace=True)  # Reset the index to avoid including it in the Dataset
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)  # Split data into train and test sets
        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)  # Ensure index is not preserved
        test_dataset = Dataset.from_pandas(test_df, preserve_index=False)  # Ensure index is not preserved
        dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
        return dataset_dict

    def generate_and_save_samples(self):
        samples = {}

        if os.path.exists(Utils.data_path(f"gpt_data_{self.num_samples_per_class}.csv")):
            df = pd.read_csv(Utils.data_path(f"gpt_data_{self.num_samples_per_class}.csv"))
            dataset_dict = self.convert_to_dataset_dict(df.to_dict(orient='records'))
        else:
            generated_data = self.generate_samples()
            df = pd.DataFrame(generated_data)  # Convert list of dictionaries to DataFrame
            df.to_csv(Utils.data_path(f"gpt_data_{self.num_samples_per_class}.csv"), index=False)
            dataset_dict = self.convert_to_dataset_dict(generated_data)
        samples[self.num_samples_per_class] = dataset_dict


        self.__ds = dataset_dict

    @property
    def ds(self):
        if self.__ds is None:
            self.generate_and_save_samples()
        return self.__ds

    @property
    def train(self):
        if self.__train is None:
            try:
                self.__train = pd.DataFrame(self.ds['train'])
            except KeyError:
                raise ValueError("Train set not found in the dataset.")
        return self.__train

    @property
    def test(self):
        if self.__test is None:
            try:
                self.__test = pd.DataFrame(self.ds['test'])
            except KeyError:
                raise ValueError("Test set not found in the dataset.")
        return self.__test
