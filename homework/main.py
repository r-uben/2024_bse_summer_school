from src.baseline import Baseline
from src.bert import BERT
from src.setfit import SetFit
from src.zero_shot import ZeroShotClassifier


from src.data import Data
from src.utils import Utils

import matplotlib.pyplot as plt
import plotly.io as pio

import os
import pandas as pd
import numpy as np
from random import randint

"""
# Description

## Objective
Classification with various techniques on this dataset.

## Dataset
Link: https://huggingface.co/datasets/dair-ai/emotion?library=datasets

## Tasks
1. **Baseline Implementation**: 
   - Train a tf-idf + logistic regression classifier with 5%, 10%, 25%, 50%, and 100% of the data.
     - Plot the learning curve (f1/precision/recall) based on the volume of data.
2. **BERT Implementation**: 
   - Train a BERT classifier with 5%, 10%, 25%, 50%, and 100% of the data.
   - Add its learning curve (f1/precision/recall) to the previous plot.
3. **BERT Model with Limited Data**: 
   - Train a setfit model using only 32 labeled examples and assess its performance.
   - Add a horizontal line on the previous plot.
4. **Zero Shot Technique**: 
   - Apply a large language model in a zero-shot learning setup with an LLM such as chatGPT.
   - If you can't apply it on all the data, a small sample should suffice.
   - Add a horizontal line on the previous plot.
5. **Generate New Data from Scratch**: 
   - Use the LLM to generate a few samples for each class (10, 50, 100).
   - Recreate the learning curve and add the performances to the previous plot.
6. **Bonus Question**: 
   - Examine some differences in what the models have learned.

## Helpful Link
This link might be useful for interacting with the LLM: [Native JSON Output from GPT-4](https://yonom.substack.com/p/native-json-output-from-gpt-4). It explains how to ask the model to provide information in a JSON format, which will be easier to organize.
"""


class Tasks(Data):

    def __init__(self):
        self.__RELOAD = False


    def get_RELOAD(self):
        return self.__RELOAD

    def set_RELOAD(self, RELOAD):
        self.__RELOAD = RELOAD

    def baseline(self):
        if (not os.path.exists(Utils.data_path("baseline.csv"))) or (self.get_RELOAD() is True):
            baseline = Baseline(
                data = Data(),
                seed = 100,
                test_size = 0.20,
                num_percentiles = 20
        )

            baseline_results = baseline.results
            baseline_results.to_csv(Utils.data_path("baseline.csv"), index=False)
        else:
            baseline_results = pd.read_csv(Utils.data_path("baseline.csv"))
        return baseline_results


    def bert(self):
        if (not os.path.exists(Utils.data_path("bert.csv"))) or (self.get_RELOAD() is True):
            bert = BERT(
                data = Data(),
            )
            bert.results.to_csv(Utils.data_path("bert.csv"), index=False)
            bert_results = bert.results
        else:
            bert_results = pd.read_csv(Utils.data_path("bert.csv"))
        return bert_results


    def setfit(self, num_samples):
        filename = f"setfit_num_samples={num_samples}.csv"
        if (not os.path.exists(Utils.data_path(filename))) or (self.get_RELOAD() is True):
            setfit = SetFit(
                data = Data(),
                num_samples=num_samples
            )
            setfit.results.to_csv(Utils.data_path(filename), index=False)
            setfit_results = setfit.results
        else:
            setfit_results = pd.read_csv(Utils.data_path(filename))

        return setfit_results

    def zero_shot(self, num_samples, temperature):

        def _prompt_function(item, few_shot_examples=None):
            prompt = {
                "instruction": "Classify the following text into one of these emotions: sadness, joy, love, anger, fear, surprise.",
                "notice": "Notice that the labels are 0, 1, 2, 3, 4, 5 respectively, each for the corresponding emotion. The equivalences are given in the following list.",
                "equivalences": {
                    "0": "sadness",
                    "1": "joy",
                    "2": "love",
                    "3": "anger",
                    "4": "fear",
                    "5": "surprise"
                },
                "response_instruction": "Only respond with the emotion label with the equivalent number, nothing else."
            }
            prompt_str = "\n\n".join([f"{key}: {value}" for key, value in prompt.items()])
            
            if few_shot_examples is not None and not few_shot_examples.empty:
                prompt_str += "\n\nHere are some examples:\n"
                for _, example in few_shot_examples.iterrows():
                    prompt_str += f"Text: {example['text']}\nEmotion: {example['label']}\n\n"
            
            prompt_str += f"\nText: {item['text']}\n\nEmotion:"
            return prompt_str
        
        filename = f"zeroshot_num_samples={num_samples}.csv"
        if (not os.path.exists(Utils.data_path(filename))) or (self.get_RELOAD() is True):
            zeroshot = ZeroShotClassifier(
                data = Data(),
                prompt_function=_prompt_function,
                num_samples=num_samples,
                temperature=temperature
            )
            zeroshot.results.to_csv(Utils.data_path(filename), index=False)
            zeroshot_results = zeroshot.results
        else:
            zeroshot_results = pd.read_csv(Utils.data_path(filename))

        return zeroshot_results



if __name__ == "__main__":

    Utils.data_visualisation(Data())
    tasks = Tasks()


    ### TASK 1: BASELINE IMPLEMENTATION
    fig_name = "metrics.png"

    baseline_results = tasks.baseline()
    fig = Utils.plot_metrics_comparison(baseline_results, label = "Baseline", color='blue', legend=False)
    fig.savefig(Utils.fig_path(fig_name), format="png")


    ### TASK 2: BERT IMPLEMENTATION

    bert_results = tasks.bert()
    fig = Utils.plot_metrics_comparison(bert_results, fig=fig, label="BERT", color='orange', legend=False)
    fig.savefig(Utils.fig_path(fig_name), format="png")


    ### TASK 3: SETFIT
    tasks.set_RELOAD(False)
    num_samples = 32
    setfit_results = tasks.setfit(num_samples)
    fig = Utils.plot_metrics_comparison(setfit_results, fig=fig, label=f"SetFit (num_samples={num_samples})", color='green', legend=False)
    fig.savefig(Utils.fig_path(fig_name), format="png")

    num_samples = 50
    setfit_results = tasks.setfit(num_samples)
    fig = Utils.plot_metrics_comparison(setfit_results, fig=fig, label=f"SetFit (num_samples={num_samples})", color='darkgreen', legend=False)
    fig.savefig(Utils.fig_path(fig_name), format="png")


    ### TASK 4: ZERO-SHOT IMPLEMENTATION


    tasks.set_RELOAD(False)
    num_samples = 32
    temperature = 0.2
    zeroshot_results = tasks.zero_shot(num_samples, temperature)

    fig = Utils.plot_metrics_comparison(zeroshot_results, fig=fig, label=f"Zero-Shot (num_samples={num_samples}_temperature={temperature})", color='purple', legend=True)
    fig.savefig(Utils.fig_path(fig_name), format="png")

    num_samples = 50
    zeroshot_results = tasks.zero_shot(num_samples, temperature)

    fig = Utils.plot_metrics_comparison(zeroshot_results, fig=fig, label=f"Zero-Shot (num_samples={num_samples}_temperature={temperature})", color='darkviolet', legend=True)
    fig.savefig(Utils.fig_path(fig_name), format="png")


    ### TASK 5: GENERATE SAMPLE