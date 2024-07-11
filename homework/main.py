from src.baseline import Baseline
from src.bert import BERT
from src.setfit import SetFit
from src.zero_shot import ZeroShotClassifier


from src.data import Data
from src.paths import Paths
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

def fig_path(fig_name):
    return os.path.join(Paths().fig, fig_name)

def data_path(data_name):
    return os.path.join(Paths().data, data_name)


def set_homework_path():
    paths = Paths()
    homework_path = paths.homework
    os.chdir(homework_path)


def data_visualisation():
    fig1 = Data().plot_word_clouds()
    fig1_name = "word_clouds.png"
    fig1.savefig(fig_path(fig1_name), format="png")
    
    fig2 = Data().plot_label_distribution()
    fig2_name = "label_distribution.png"
    fig2.savefig(fig_path(fig2_name), format="png")


def plot_metrics(df, fig=None, label=None, legend=True, color=None):
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    
    # Normalize percentage values if they're not already in the 0-100 range
    if df['percentage'].max() <= 1:
        df['percentage'] = df['percentage'] * 100
    
    # Sort the dataframe by percentage
    df = df.sort_values('percentage')
    
    if fig is None:
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    else:
        axs = fig.get_axes()
    
    axs = np.array(axs).ravel()

    M = randint(1,10)
    markers = ['o', 'x']
    j = randint(0,1)
    for i, metric in enumerate(metrics):
        metric_data = df[df['metric'] == metric]
        if len(metric_data) == 1:
 
            axs[i].plot(np.linspace(0,100,M), [metric_data['value'].values[0]] * M, color=color, marker=markers[j], linestyle='-', label=label)
        else:
            axs[i].plot(metric_data['percentage'], metric_data['value'], marker='o', label=label, color=color)
        axs[i].set_title(f'{metric.capitalize()}')
        axs[i].set_xlabel('Percentage')
        axs[i].set_ylabel(metric.capitalize())
        if legend and i == 0:  # Only add legend to the first subplot
            axs[i].legend()
        axs[i].grid(True, linewidth=0.5)
        axs[i].set_xlim(0, 100)  # Set x-axis from 0 to 100
        for spine in ['top', 'right', 'left', 'bottom']:
            axs[i].spines[spine].set_visible(False)

    plt.tight_layout()
    return fig


if __name__ == "__main__":

    data_visualisation()


    ### TASK 1: BASELINE IMPLEMENTATION
    fig_name = "metrics.png"

    if not os.path.exists(data_path("baseline.csv")):
        baseline = Baseline(
            data = Data(),
            seed = 100,
            test_size = 0.20,
            num_percentiles = 20
    )

        baseline_results = baseline.results
        baseline_results.to_csv(data_path("baseline.csv"), index=False)
    else:
        baseline_results = pd.read_csv(data_path("baseline.csv"))

    fig = plot_metrics(baseline_results, label = "Baseline", color='blue', legend=False)

    fig.savefig(fig_path(fig_name), format="png")


    ### TASK 2: BERT IMPLEMENTATION

    if not os.path.exists(data_path("bert.csv")):
        bert = BERT(
            data = Data(),
        )
        bert.results.to_csv(data_path("bert.csv"), index=False)
        bert_results = bert.results
    else:
        bert_results = pd.read_csv(data_path("bert.csv"))

    fig = plot_metrics(bert_results, fig=fig, label="BERT", color='orange', legend=False)
    fig.savefig(fig_path(fig_name), format="png")


    ### TASK 3: SETFIT
    RELOAD = True
    if (not os.path.exists(data_path("setfit.csv"))) or (RELOAD is True):
        setfit = SetFit(
            data = Data(),
        )
        setfit.results.to_csv(data_path("setfit.csv"), index=False)
        setfit_results = setfit.results
    else:
        setfit_results = pd.read_csv(data_path("setfit.csv"))

    fig = plot_metrics(setfit_results, fig=fig, label="SetFit", color='green', legend=True)
    fig.savefig(fig_path(fig_name), format="png")

    ### TASK 4: ZERO-SHOT IMPLEMENTATION
    def _prompt_function(item, few_shot_examples=None):
        prompt = f"""
            Classify the following text into one of these emotions: sadness, joy, love, anger, fear, surprise.
            Notice that the labels are 0, 1, 2, 3, 4, 5 respectively, each for the corresponding emotion. The equivalences are given in the
            following list.
            0 = sadness
            1 = joy
            2 = love
            3 = anger
            4 = fear
            5 = surprise.
                Only respond with the emotion label with the equivalent number, nothing else.
            """
        if few_shot_examples is not None and not few_shot_examples.empty:
            prompt += "\n\nHere are some examples:\n"
            for _, example in few_shot_examples.iterrows():
                prompt += f"Text: {example['text']}\nEmotion: {example['label']}\n\n"
        
        prompt += f"\nText: {item['text']}\n\nEmotion:"
        return prompt

    RELOAD = False
    if (not os.path.exists(data_path("zeroshot.csv"))) or (RELOAD is True):
        zeroshotGPT = ZeroShotClassifier(
            Data(), prompt_function=_prompt_function)
        zeroshotGPT.results.to_csv(data_path("zeroshot.csv"), index=False)
        zeroshotGPT_results = zeroshotGPT.results
    else:
        zeroshotGPT_results = pd.read_csv(data_path("zeroshot.csv"))

    fig = plot_metrics(zeroshotGPT_results, fig=fig, label="Zero-Shot", color='purple', legend=True)
    fig.savefig(fig_path(fig_name), format="png")


    ### TASK 5: GENERATE SAMPLE
#
#
#