from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.utils import Utils
from tqdm import tqdm
import pandas as pd

class Baseline:

    def __init__(self, data, max_iter=100, num_percentiles=20, seed=100, test_size=0.20):
        self.data = data
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression(max_iter=max_iter)
        self.__seed = seed
        self.__num_percentiles = num_percentiles
        self.__test_size = test_size
        self.__results = None

    @property
    def percentages(self):
        return [i / self.__num_percentiles for i in range(1, 21, 4)]

    @property
    def results(self):
        if self.__results is None:
            self.__results = self.get_results()
        return self.__results
    
    def get_results(self, data=None, model=None):
        if data is None:
            data = self.data
        if model is None:
            model = self.model
        results = {}
        for percentage in tqdm(self.percentages, desc="Processing percentages"):
            train_subset = data.train.sample(frac=percentage, random_state=self.__seed)
            X_train, X_val, y_train, y_val = Utils.get_train_sample(train_subset, self.__test_size, self.__seed)
            X_train = self.vectorizer.fit_transform(X_train)
            X_val = self.vectorizer.transform(X_val)
            Utils.fit(model, X_train, y_train)
            y_pred = Utils.predict(model, X_val)
            results[percentage] = Utils.metrics(y_val, y_pred)
        return Utils.convert_results_to_dataframe(results)