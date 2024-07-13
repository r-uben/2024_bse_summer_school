from src.utils import Utils

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from tqdm import tqdm


import pandas as pd
import torch

class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class Classifier:
    def __init__(self, epochs=10, batch_size=16):
        # The model: BERT
        self.__model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
        self.__device = Utils.get_device()
        self.__model.to(self.__device)

        # The optimizer: AdamW
        self.__optimizer = AdamW(self.__model.parameters(), lr=5e-5)

        # 
        self.__batch_size = batch_size
        self.__epochs = epochs

    @property
    def model(self):
        return self.__model
    
    @property
    def device(self):
        return self.__device
    
    def train(self, train_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.__batch_size, shuffle=True)

        self.__model.train()
        for epoch in range(self.__epochs):
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            for batch in progress_bar:
                self.__optimizer.zero_grad()
                inputs = {key: val.to(self.__device) for key, val in batch.items()}
                outputs = self.__model(**inputs)
                loss = outputs.loss
                loss.backward()
                self.__optimizer.step()
                progress_bar.set_postfix(loss=loss.item())
    

class BERT:
    
    def __init__(self, data, num_percentiles=20, epochs=10, batch_size=16):

        self.__data = data
        # The tokenizer: BERT
        self.__tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.__num_percentiles = num_percentiles
        #
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__num_percentiles = num_percentiles

        self.__classifier = Classifier(self.__epochs, self.__batch_size)
        
        self.__results = None

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
        for percentage in tqdm(self.percentages, desc="Processing percentages"):
            train_encodings, train_labels = Utils.preprocess_data(self.__data.train, self.__tokenizer, percentage)
            train_dataset = EmotionDataset(train_encodings, train_labels)
            self.__classifier.train(train_dataset)

            test_encodings, test_labels = Utils.preprocess_data(self.__data.test, self.__tokenizer)
            test_dataset = EmotionDataset(test_encodings, test_labels)
            results[percentage] = self.evaluate(self.__classifier, test_dataset, self.__batch_size)
        
        return Utils.convert_results_to_dataframe(results)

    def evaluate(self, model, dataset, batch_size=16):
        model.model.eval()
        dataloader = DataLoader(dataset, batch_size, shuffle=False)
        predictions, true_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                inputs = {key: val.to(model.device) for key, val in batch.items() if key != 'labels'}
                outputs = model.model(**inputs)
                logits = outputs.logits
                predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())

        return Utils.metrics(true_labels, predictions)