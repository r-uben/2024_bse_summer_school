

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer

import nltk
import numpy as np
import pandas as pd
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