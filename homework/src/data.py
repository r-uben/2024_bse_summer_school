from collections import Counter
from datasets import load_dataset
from wordcloud import WordCloud
from nltk.corpus import stopwords

import nltk


import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class Data: 

    def __init__(self, ds=None):
       self.__ds = ds
       self.__train = None
       self.__test = None

       if not nltk.data.find('corpora/stopwords.zip'):
           nltk.download('stopwords')


    @property
    def default_stopwords(self):
        return set(stopwords.words('english'))

    @property
    def label_names(self):
        return {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}


    @property 
    def ds(self):
        if self.__ds is None:
            self.__ds = load_dataset("dair-ai/emotion", "split")
        else:
            return self.__ds
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
    
    def add_stopwords(self, stopwords):
        return self.default_stopwords.union(set(stopwords))
    

    def get_optimized_stopwords(self, frequency_threshold=0.001):
        # Start with NLTK's English stopwords
        stop_words = set(stopwords.words('english'))

        # Add common contractions and their expansions
        contractions = {"'s", "'re", "'ve", "'ll", "'d", "n't"}
        stop_words.update(contractions)

        # Count words across all emotions
        all_words = Counter()
        total_words = 0
        for label in range(6):
            texts = ' '.join(self.train[self.train['label'] == label]['text']).lower().split()
            all_words.update(texts)
            total_words += len(texts)

        # Add very frequent words to stopwords
        for word, count in all_words.items():
            if count / total_words > frequency_threshold:
                stop_words.add(word)

        return stop_words

    def plot_word_clouds(self, additional_stopwords=None, frequency_threshold=0.001):

        # Get optimized stopwords
        stop_words = self.get_optimized_stopwords(frequency_threshold)

        # Add additional stopwords if provided
        if additional_stopwords:
            stop_words.update(additional_stopwords)

        # Create a 2x3 subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Word Clouds for Each Emotion", fontsize=24)

        for i, label in enumerate(range(6)):
            row = i // 3
            col = i % 3

            # Filter texts for the current label
            texts = ' '.join(self.train[self.train['label'] == label]['text'])
            
            # Generate word cloud with stopwords
            wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                  stopwords=stop_words, min_font_size=10, 
                                  max_words=100, collocations=False).generate(texts)
            
            # Plot the word cloud
            axes[row, col].imshow(wordcloud, interpolation='bilinear')
            axes[row, col].set_title(self.label_names[label])
            axes[row, col].axis('off')

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
    def plot_label_distribution(self):
        # Count the occurrences of each label
        label_counts = Counter(self.train['label'])

        # Create a bar plot using matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(label_counts.keys(), label_counts.values())

        ax.set_title('Distribution of Labels in the Training Set')
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')

        return fig