"""
This module contains CampaignTextModel class for training base/center model
NOTE: FOR NOW, IN ORDER TO ADD SUMMARIZED_TEXT TO MODEL FEATURES WE NEED TO ADD OCR AS WELL.
"""
from transformers import AutoTokenizer, EarlyStoppingCallback
from datasets import load_dataset
from datasets import DatasetDict
from datasets import Dataset
from datasets import Features
from datasets import Value
from datasets import ClassLabel
from datasets import Sequence
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_metric
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
BASE = "BASE"
CENTER = "CENTER"
'''def train(self, learning_rate=2e-5, per_device_train_batch_size=30, per_device_eval_batch_size=30, epochs=5,
              decay=0.01)'''
directions = ["CENTER","BOTH","BASE"]

class CampaignTextModel:

    def __init__(self, lang_model_name: str, direction, path_name: str,test_size: float,
                 data_filter_func,test_name_filter_func,additional_features_dict = {}):
        self.lang_model_name = lang_model_name
        self.direction = direction
        self.trainer = None
        self.dataset = None
        self.db_size = 0 # not initialized yet
        self.path_name = path_name
        self.test_size = test_size
        self.data_filter_func = data_filter_func
        self.test_name_filter_func = test_name_filter_func
        self.additional_features_dict = additional_features_dict

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        metric = load_metric("accuracy")
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    def create_filtered_dataset_by_filter_names(self,filter_names):
        filtered_df = pd.DataFrame(filter_names,columns=['name'])
        filtered_df = filtered_df.merge(self.df,on='name')
        filtered_df.dropna(inplace=True)
        del filtered_df['name']
        return Dataset.from_pandas(filtered_df,features=self.features,preserve_index=False)
    def split_names_by_test_filter_and_test_size(self):
        if (self.test_name_filter_func != None):
            non_test_names = self.names[~self.names.apply(self.test_name_filter_func)]
            test_names = self.names[self.names.apply(self.test_name_filter_func)][:int(self.db_size*(self.test_size/2))]
        else:
            non_test_names = self.names[:int(self.db_size*(1-self.test_size/2))]
            test_names = self.names[int(self.db_size*(1-self.test_size/2)):]
        return (non_test_names, test_names)
    def load_data_from_csv(self):
        # Define features
        class_names = ["0", "1","2"] if self.direction == None else ["0","1"]
        self.class_names = class_names
        self.features_dict = {'text': Value(dtype='string'), 'label': ClassLabel(names=class_names)}
        self.features_dict.update(self.additional_features_dict)
        self.features = Features(self.features_dict)
        # Load dataset from csv
        self.df = pd.read_csv(self.path_name)
        # Filter data by data_filter_func if it's not None
        if (self.data_filter_func != None):
            self.df = self.df[self.df.apply(self.data_filter_func,axis=1)]
        # Save unique names (don't use drop_duplicates because there are videos with the same name but different text!)
        self.names = pd.Series(self.df['name'].unique())
        # Consider db_size as the number of unique names
        self.db_size = len(self.names)
        # Shuffle names to garuantee ramdom split of the data to train-valid-test
        np.random.shuffle(self.names)
        # Split names by test_filter_func if it's not None and by test_size
        non_test_names, test_names = self.split_names_by_test_filter_and_test_size()
        # Split non_test_names to train-valid by test_size
        train_names = non_test_names[:int(self.db_size*(1-self.test_size))]
        valid_names = non_test_names[int(self.db_size*(1-self.test_size)):int(self.db_size*(1-self.test_size/2))]
        # Create train-valid-test datasets
        train_dataset = self.create_filtered_dataset_by_filter_names(train_names)
        valid_dataset = self.create_filtered_dataset_by_filter_names(valid_names)
        test_dataset = self.create_filtered_dataset_by_filter_names(test_names)
        # Create dataset which wraps train-valid-test datasets as a DatasetDict
        self.dataset = DatasetDict({
            "train": train_dataset,
            "valid": valid_dataset,
            "test": test_dataset
        })
    def train(self, learning_rate=2e-5, per_device_train_batch_size=30, per_device_eval_batch_size=30, epochs=5,
              decay=0.01,early_stopping_patience=3):
        # Tokenizer tuning
        def my_tokenizer(examples,tokenizer): #return tokenizer
            print(examples,type(examples))
            return tokenizer([examples[text_feature] for text_feature in self.text_features_list],padding=True, truncation=True)
        tokenizer = AutoTokenizer.from_pretrained(self.lang_model_name) #bert tokenizer
        self.tokenizer = tokenizer
        self.text_features_list = [k for k in self.additional_features_dict if self.additional_features_dict[k].dtype == 'String']
        if (len(self.text_features_list)==0):
            tokenized_ds = self.dataset.map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)
        elif (len(self.text_features_list)==1):
            tokenized_ds = self.dataset.map(lambda examples: tokenizer(examples["text"],examples[self.text_features_list[0]], truncation=True), batched=True)
        else:
            self.text_features_list.append('text')
            tokenized_ds = self.dataset.map(lambda examples: my_tokenizer(examples,tokenizer), batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(self.lang_model_name, num_labels=2+(self.direction==None))
        # Trainer parameters
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=epochs,
            weight_decay=decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=5,
            load_best_model_at_end = True
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds["train"].shuffle(),
            eval_dataset=tokenized_ds["valid"].shuffle(),
            tokenizer=tokenizer,
            compute_metrics=CampaignTextModel.compute_metrics,
            data_collator=data_collator,
            callbacks = [EarlyStoppingCallback(early_stopping_patience = early_stopping_patience)]
        )
        # Train model
        trainer.train()
        self.trainer = trainer       

    def predict(self, tokenized_ds, return_number=False):
        print('tokenized_ds')
        print(tokenized_ds)
        predictions = self.trainer.predict(tokenized_ds)
        print('predictions.predictions')
        print(predictions.predictions)
        pred_num = np.argmax(predictions.predictions, axis=-1)
        print('pred_num')
        print(pred_num)
        if return_number:
            return pred_num
        if self.direction == None:
            return directions[pred_num]
        return self.direction if pred_num else "NOT_" + self.direction

    def predict_by_text(self, text_string: str, return_number=False, isDataSet = True):
        # Build dataset with one row
        print(text_string)
        data_to_predict = Dataset.from_dict({"text": [text_string]})
        print('data_to_predict')  
        print(data_to_predict.data)
        tokenized_ds = data_to_predict.map(lambda examples: self.tokenizer(examples["text"], truncation=True), batched=False)
        print('tokenized_ds')
        print(tokenized_ds)
        predictions = self.trainer.predict(tokenized_ds)
        print('predictions.predictions')
        print(predictions.predictions)
        pred_num = np.argmax(predictions.predictions, axis=-1)
        print('pred_num')
        print(pred_num)
        if return_number:
            return pred_num
        if self.direction == None:
            return directions[pred_num]
        return self.direction if pred_num else "NOT_" + self.direction
  
    def predict_by_dataset(self, ds: Dataset, return_number=False, isDataSet = True):
        tokenized_ds = ds.map(lambda examples: self.tokenizer(examples["text"], truncation=True), batched=False)
        print("tokenized ds")
        print(tokenized_ds)
        predictions = self.trainer.predict(tokenized_ds)
        pred = np.argmax(predictions.predictions, axis = -1)
        print("pred")
        print(pred)
        return pred
