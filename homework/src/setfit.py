from src.data import Data
from src.paths import Paths
from src.utils import Utils
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset


import pandas as pd

# This class is created to address an incompatibility issue between SetFit and Transformers
# specifically related to the eval_strategy input.
class CompatibleTrainingArguments(TrainingArguments):
    @property
    def eval_strategy(self):
        return self.evaluation_strategy
    

class Classifier:
    def __init__(self, train_data: Data, 
                 batch_size = 16,
                 num_epochs = 10,
                 num_iterations = 20):
        
        self.__train_data = train_data

        self.__batch_size = batch_size
        self.__num_epochs = num_epochs
        self.__num_iterations = num_iterations
        ###
        # Initialize the SetFit model using a pre-trained sentence transformer model
        self.model = SetFitModel.from_pretrained(
            "sentence-transformers/paraphrase-mpnet-base-v2"
        )
        self.__args = None

        ### 
        self.__train_dataset = None

    
    def train(self, train_dataset, num_samples):
        few_shot_dataset = sample_dataset(train_dataset, label_column="label", num_samples=num_samples)
        len(few_shot_dataset)
        trainer = self.get_trainer(few_shot_dataset)
        trainer.train()

    def get_trainer(self, train_dataset):
        trainer = Trainer(
            model=self.model,
            args=self.get_args(),
            train_dataset=train_dataset,
        )
        return trainer

    
    def get_args(self):
        if self.__args is None:
            self.__args  = \
                CompatibleTrainingArguments(
                    output_dir=Paths().results,
                    batch_size=self.__batch_size,
                    num_epochs=self.__num_epochs,
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    load_best_model_at_end=True,
                    num_iterations=self.__num_iterations,
            )
        return self.__args


    def get_train_dataset(self):
        if self.__train_dataset is None:
            self.__train_dataset = Dataset.from_dict({
                "text": self.train_sample,
                "label": self.train_labels
            })
        return self.__train_dataset




class SetFit():

    def __init__(self, 
                 data: Data, 
                 num_percentiles = 20,
                 num_samples = 32,
                 batch_size = 16,
                 num_epochs = 10,
                 num_iterations = 20):
    

        self.__data = data
        self.__classifier = Classifier(data.train, batch_size, num_epochs, num_iterations)
        self.__num_percentiles = num_percentiles
        self.__num_samples = num_samples

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
        #for percentage in self.percentages:
        percentage = 0.5
        split_sample = Utils.get_split_sample(self.__data.train, percentage)
        percentage = self.__num_samples / int(len(self.__data.train))

        train_dict = {'text': split_sample['train']['text'], 'label': split_sample['train']['label']}
        train_dataset = Dataset.from_dict(train_dict)
        self.__classifier.train(train_dataset=train_dataset, num_samples=self.__num_samples)
        
        # Add error handling
        try:
            valid_preds = self.__classifier.model.predict(split_sample['valid']['text'])
            metrics = Utils.metrics(split_sample['valid']['label'], valid_preds)
            results[percentage] = metrics
        except Exception as e:
            print(f"Error predicting for percentage {percentage}: {str(e)}")
    
        return Utils.convert_results_to_dataframe(results)
    



            

        
        





    # @property
    # def split_sample(self):
    #     if self.__split_sample is None:
    #        self.__split_sample = Utils.get_split_sample(self.__train_data)
    #     return self.__split_sample

    # @property
    # def train_sample(self):
    #     if self.__train_sample is None:
    #         self.__train_sample = self.split_sample[0]
    #     return self.__train_sample
    
    # @property
    # def valid_sample(self):
    #     if self.__valid_sample is None:
    #         self.__valid_sample = self.split_sample[1]
    #     return self.__valid_sample
    
    # @property
    # def train_labels(self):
    #     if self.__train_labels is None:
    #         self.__train_labels = self.split_sample[2]
    #     return self.__train_labels
    
    # @property
    # def valid_labels(self):
    #     if self.__valid_labels is None:
    #         self.__valid_labels = self.split_sample[3]
    #     return self.__valid_labels
    
    # @property
    # def train_dataset(self):
    #     if self.__train_dataset is None:
    #         self.__train_dataset = self.get_train_dataset()
    #     return self.__train_dataset