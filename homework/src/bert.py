import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from src.utils import Utils
import pandas as pd

class BERT:

    def __init__(self, data, 
                 model_name='bert-base-uncased',
                 num_labels=6,  # Adjust based on your dataset
                 num_epochs=10,
                 batch_size=32,
                 learning_rate=2e-5,
                 num_percentiles=20,
                 seed=100,
                 test_size=0.20):
        
        self.data = data
        # Use MPS if available, otherwise fall back to CPU
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.__models = {}
        self.model_name = model_name
        self.num_labels = num_labels
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.__seed = seed
        self.__num_percentiles = num_percentiles
        self.__test_size = test_size
        self.__results = None

        # Create a fixed validation set
        self.X_train, self.X_val, y_train, y_val = Utils.get_train_sample(self.data.train, self.__test_size, self.__seed)
        self.y_train = y_train.tolist() if hasattr(y_train, 'tolist') else y_train
        self.y_val = y_val.tolist() if hasattr(y_val, 'tolist') else y_val


    @property
    def models(self):
        return self.__models
    

    @property
    def percentages(self):
        return [i/self.__num_percentiles for i in range(1, 21, 4)]

    @property
    def results(self):
        if self.__results is None:
            results = self.get_results()
            results_df = pd.DataFrame.from_dict(results, orient='index')
            results_df = results_df.reset_index().rename(columns={'index': 'percentage'})
            results_df['percentage'] = results_df['percentage'] * 100
            results_df = results_df.melt(id_vars=['percentage'], var_name='metric', value_name='value')
            results_df = results_df.sort_values(['percentage', 'metric'])
            results_df = results_df.reset_index(drop=True)
            self.__results = results_df
        return self.__results

    def fit(self, train_dataloader, val_dataloader):
        model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels).to(self.device)
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * self.num_epochs)
        
        best_val_loss = float('inf')
        patience = 3
        no_improve = 0
        
        for epoch in range(self.num_epochs):
            model.train()
            for batch in train_dataloader:
                optimizer.zero_grad()
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()

            val_loss = self.evaluate(model, val_dataloader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                best_model = model.state_dict()
            else:
                no_improve += 1
            
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        model.load_state_dict(best_model)
        return model

    def evaluate(self, model, dataloader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
        return total_loss / len(dataloader)

    def predict(self, model, dataloader):
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().tolist())
        return predictions

    def prepare_data(self, texts, labels):
        # Convert texts to a list if it's not already
        if not isinstance(texts, list):
            texts = texts.tolist()
        
        # Ensure all elements are strings
        texts = [str(text) for text in texts]
        
        # Convert labels to a list if it's not already
        if not isinstance(labels, list):
            labels = labels.tolist()
        
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=512)
        dataset = TensorDataset(
            torch.tensor(encodings['input_ids']),
            torch.tensor(encodings['attention_mask']),
            torch.tensor(labels)
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def get_results(self, data=None, model=None):
        if data is None:
            data = self.data
        results = {}
        val_dataloader = self.prepare_data(self.X_val, self.y_val)
        if model is None:
        
            for percentage in tqdm(self.percentages, desc="Processing percentages"):
                train_subset = data.train.sample(frac=percentage, random_state=self.__seed)
                X_train_subset = train_subset['text']
                y_train_subset = train_subset['label'].tolist()
                
                train_dataloader = self.prepare_data(X_train_subset, y_train_subset)
                
                self.__models[percentage] = self.fit(train_dataloader, val_dataloader)
                model = self.__models[percentage]

                y_pred = self.predict(model, val_dataloader)
                results[percentage] = Utils.metrics(self.y_val, y_pred)
        else:
            val_dataloader = self.prepare_data(data.train['text'], data.train['label'])
            y_pred = self.predict(model, val_dataloader)
            results[percentage] = Utils.metrics(self.y_val, y_pred)

        return results