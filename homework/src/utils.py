from botocore.exceptions import ClientError
from nltk.corpus import stopwords
from random import randint

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer

from src.paths import Paths

import boto3

import nltk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import spacy
import re
#this requires spacy to be installed
sp = spacy.load('en_core_web_sm')

#getting a library of stopwords and defining a lemmatizer
porter= SnowballStemmer("english")
lmtzr = WordNetLemmatizer()


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


class Utils:

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    @staticmethod
    def metrics(y_true, y_pred):
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted')
        accuracy = accuracy_score(y_true, y_pred)
        return {'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': accuracy}

    @staticmethod
    def convert_results_to_dataframe(results):
        df = pd.DataFrame(results).T
        df.reset_index(inplace=True)
        df = df.melt(id_vars=['index'], var_name='metric', value_name='value')
        df.rename(columns={'index': 'percentage'}, inplace=True)
        return df

    @staticmethod
    def get_split_sample(data, percentage):
        train_data, valid_data = train_test_split(data, test_size=percentage)
        train_texts = train_data['text'].tolist()
        train_labels = train_data['label'].tolist()
        valid_texts = valid_data['text'].tolist()
        valid_labels = valid_data['label'].tolist()
        return {'train': {'text': train_texts, 'label': train_labels},
                'valid': {'text': valid_texts, 'label': valid_labels},
                'train_data': train_data,
                'valid_data': valid_data}

    @staticmethod
    def get_secret(secret_name, region_name):
        client = boto3.client('secretsmanager', region_name=region_name)
        try:
            response = client.get_secret_value(SecretId=secret_name)
            secret = response.get('SecretString')
            if secret is None:
                secret = response['SecretBinary'].decode('utf-8')
            return secret  # Ensure it returns a JSON string
        except ClientError as e:
            raise e

    def preprocess_text(self, text, modulation=1, join=True):
        # Tokenize
        tokens = self.tokenize(text.lower(), modulation=modulation)
        # Remove stopwords and non-alphabetic tokens
        tokens = [token for token in tokens 
                  if token.isalpha() 
                  and token not in self.stop_words]
        if join:
            return ' '.join(tokens)
        else:
            return tokens

    def strip(self, word):
        mod_string = re.sub(r'\W+', '', word)
        return mod_string
    
    #the following leaves in place two or more capital letters in a row
    #will be ignored when using standard stemming
    def abbr_or_lower(self, word):
        # If the word is an abbreviation, return it as is, e.g., NASA
        if re.match('([A-Z]+[a-z]*){2,}', word):
            return word
        else:
            return word.lower()

    def tokenize(self, text, modulation):
        if modulation<2:
            tokens = re.split(r'\W+', text)
            stems = []
            # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
            for token in tokens:
                lowers=self.abbr_or_lower(token)
                if lowers not in self.stop_words:
                    if re.search('[a-zA-Z]', lowers): # Filter numeric tokens
                        if modulation==0:
                            stems.append(lowers)
                        if modulation==1:
                            stems.append(porter.stem(lowers))
        else:
            sp_text=sp(text)
            stems = []
            lemmatized_text=[]
            for word in sp_text:
                lemmatized_text.append(word.lemma_)
            stems = [self.abbr_or_lower(self.strip(w)) 
                     for w in lemmatized_text 
                     if self.abbr_or_lower(self.strip(w))
                     not in self.stop_words]
        return stems

    def vectorize(self, tokens, vocab):
        vector=[]
        for w in vocab:
            vector.append(tokens.count(w))
        return vector
    


    @staticmethod   
    def fit(model, X, y):
        model.fit(X, y)

    @staticmethod
    def predict(model, X):
        return model.predict(X)

    @staticmethod
    def metrics(y_true, y_pred):
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted')
        accuracy = accuracy_score(y_true, y_pred)
        return {'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': accuracy}

    @staticmethod
    def get_train_sample(train_subset, test_size, seed):
        X_train, X_val, y_train, y_val = train_test_split(
            train_subset['text'],
            train_subset['label'],
            test_size=test_size,
            random_state=seed
        )
        return X_train, X_val, y_train, y_val


    @staticmethod
    def plot_metrics(results):
        fig = go.Figure()

        for metric in results.metric.unique():
            df =results[results.metric==metric]
            fig.add_trace(go.Scatter(
                x=df['percentage'],
            y=df['value'],
            mode='lines+markers',
                name=metric
            ))

        return fig
    

    @staticmethod
    def fig_path(fig_name):
        return os.path.join(Paths().fig, fig_name)

    @staticmethod
    def data_path(data_name):
        return os.path.join(Paths().data, data_name)

    @staticmethod
    def set_homework_path():
        paths = Paths()
        homework_path = paths.homework
        os.chdir(homework_path)

    @staticmethod
    def data_visualisation(data):
        fig1 =data.plot_word_clouds()
        fig1_name = "word_clouds.png"
        fig1.savefig(Utils.fig_path(fig1_name), format="png")
        
        fig2 = data.plot_label_distribution()
        fig2_name = "label_distribution.png"
        fig2.savefig(Utils.fig_path(fig2_name), format="png")


    @staticmethod
    def plot_metrics_comparison(df, fig=None, label=None, legend=True, color=None):
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

        fig.tight_layout()
        return fig


