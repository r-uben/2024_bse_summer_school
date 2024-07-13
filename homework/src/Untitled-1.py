

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split

from botocore.exceptions import NoCredentialsError, PartialCredentialsError

from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer

import boto3
import nltk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import spacy
import torch
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
    def performance_figure(results, title):

        fig = go.Figure()
        fig = Utils.plot_metrics(results)
        fig.update_layout(
            title=title,
            xaxis_title='Percentage of Training Data',
            yaxis_title='Score',
            legend_title='Metric',
            template='plotly_white'
        )

        return fig



    @staticmethod
    def get_secret(secret_name, region_name):

        # Create a Secrets Manager client
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )

        try:
            get_secret_value_response = client.get_secret_value(
                SecretId=secret_name
            )
        except (NoCredentialsError, PartialCredentialsError) as e:
            raise Exception("Credentials not available") from e

        # Decrypts secret using the associated KMS key.
        secret = get_secret_value_response['SecretString']
        return secret

     
    @staticmethod
    def convert_results_to_dataframe(results):
        data = []
        for percentage, metrics in results.items():
            for metric, value in metrics.items():
                data.append({
                    'metric': metric,
                    'value': value,
                    'percentage': percentage
                })
        return pd.DataFrame(data)
        

    @staticmethod
    def get_split_sample(data, percentage, seed=100):
        split_sample = train_test_split(
            data["text"],
            data["label"],
            test_size=percentage,
            random_state=seed
        )
        split_sample_dict = {
            "train": {'text': split_sample[0], 'label': split_sample[2]},
            "valid": {'text': split_sample[1], 'label': split_sample[3]}
        }

        return split_sample_dict

    @staticmethod
    def get_train_val_sample(train_subset, test_size, seed=100):
        X_train, X_val, y_train, y_val = train_test_split(
            train_subset['text'],
            train_subset['label'],
            test_size=test_size,
            random_state=seed
        )
        return X_train, X_val, y_train, y_val

    
    @staticmethod
    def preprocess_data(data, tokenizer, percentage=0.1):
        
        # SPLIT THE DATA
        if percentage < 1.0:
            data, _ = train_test_split(
                data,
            test_size=1-percentage,
            stratify=data['label']
            )

        # TOKENIZE THE DATA
        encodings = tokenizer(
            data['text'].tolist(), 
            truncation=True, 
            padding=True, 
            max_length=128
        )

        # GET THE LABELS
        labels = data['label'].tolist()

        return encodings, labels
    

    @staticmethod
    def get_device():
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

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
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}


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