'''trainer '''
"""
This module contains CampaignTextModel class for training base/center model
"""
!pip install transformers
!pip install datasets
from transformers import AutoTokenizer
from datasets import load_dataset
from datasets import DatasetDict
from datasets import Dataset
from datasets import Features
from datasets import Value
from datasets import ClassLabel
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_metric
import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
BASE = "BASE"
CENTER = "CENTER"
'''def train(self, learning_rate=2e-5, per_device_train_batch_size=30, per_device_eval_batch_size=30, epochs=5,
              decay=0.01)'''
directions = ["CENTER","BOTH","BASE"]

class CampaignTextModel:

    def __init__(self, lang_model_name: str, direction):
        self.lang_model_name = lang_model_name
        self.direction = direction
        self.trainer = None
        self.dataset = None

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        metric = load_metric("accuracy")
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def load_data_from_csv(self, data_path: str, test_size: float):
        # Define features
        class_names = ["0", "1","2"] if self.direction == None else ["0","1"]
        self.class_names = class_names
        features = Features({'text': Value('string'), 'label': ClassLabel(names=class_names)})
        # Load dataset from csv
        df = pd.read_csv(data_path)
        names = df['name'].unique()
        np.random.shuffle(names)
        train_names = names[:int(len(names)*(1-test_size))]
        train_df = pd.DataFrame(train_names,columns=['name'])
        train_df = train_df.merge(df,on='name')
        train_df.dropna(inplace=True)
        print('train_df')
        print(train_df[['text','label']])
        train_dataset = Dataset.from_pandas(train_df,split="train",features=features)
        valid_names = names[int(len(names)*(1-test_size)):int(len(names)*(1-test_size/2))]
        valid_df = pd.DataFrame(valid_names,columns=['name'])
        valid_df = valid_df.merge(df,on='name')
        valid_df.dropna(inplace=True)
        print('valid_df')
        print(valid_df[['text','label']])
        valid_dataset = Dataset.from_pandas(valid_df,features=features)
        test_names = names[int(len(names)*(1-test_size/2)):]
        test_df = pd.DataFrame(test_names,columns=['name'])
        test_df = test_df.merge(df,on='name')
        test_df.dropna(inplace=True)
        print('test_df')
        print(test_df[['text','label']])
        test_dataset = Dataset.from_pandas(test_df,features=features)
        #self.dataset = load_dataset("csv", data_files=data_path, split="train", features=features)
        ## Split data into train, test and validation
        #train_testvalid = self.dataset.train_test_split(test_size=test_size)
        #test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
        #self.dataset = DatasetDict({
        #    "train": train_testvalid["train"],
        #    "test": test_valid["test"],
        #    "valid": test_valid["train"]})
        self.dataset = DatasetDict({
            "train": train_dataset,
            "valid": valid_dataset,
            "test": test_dataset
        })
    def train(self, learning_rate=2e-5, per_device_train_batch_size=30, per_device_eval_batch_size=30, epochs=5,
              decay=0.01):
        # Tokenizer tuning
        tokenizer = AutoTokenizer.from_pretrained(self.lang_model_name)
        self.tokenizer = tokenizer
        tokenized_ds = self.dataset.map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)
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
            logging_steps=5,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds["train"].shuffle(),
            eval_dataset=tokenized_ds["valid"].shuffle(),
            tokenizer=tokenizer,
            compute_metrics=CampaignTextModel.compute_metrics,
            data_collator=data_collator,
        )
        # Train model
        trainer.train()
        self.trainer = trainer
        #print(trainer)       

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
        def tokenizer(examples):
            #print('#####')
            #print(examples)
            #print('#####')
            return self.tokenizer(examples["text"], truncation=True)
        tokenized_ds = data_to_predict.map(lambda examples: tokenizer(examples), batched=False)
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
        def tokenizer(examples):
            #print('#####')
            #print(examples)
            #print('#####')
            return self.tokenizer(examples["text"], truncation=True)
        tokenized_ds = ds.map(lambda examples: tokenizer(examples), batched=False)
        print("tokenized ds")
        print(tokenized_ds)
        predictions = self.trainer.predict(tokenized_ds)
        #print("predictions")
        #print(predictions)
        pred = np.argmax(predictions.predictions, axis = -1)
        print("pred")
        print(pred)
        return pred
    '''
    def calc_confusion_matrix(self, predictions, true_labels):
      confusion_matrix = np.zeros((2,2))
      for i in range(len(predictions)):
        if(predictions[i] == 1):
          if(true_labels[i] == 1):
            confusion_matrix[0][0]+=1
          else:
            confusion_matrix[0][1]+=1
        else:
          if(true_labels[i]==1):
            confusion_matrix[1][0]+=1
          else:
            confusion_matrix[1][1]+=1
      return confusion_matrix
    '''
        
if __name__ == "__main__":
    text = "The crime Medicare fraud. The victims American taxpayers. The boss Mitt Romney Romney supervised to company guilty. A massive Medicare fraud. That's a fact. 25 million dollars in unnecessary blood tests. Right under Romney's nose."
    m = CampaignTextModel("distilbert-base-uncased", None)
    m.load_data_from_csv("tags_double.csv", 0.4)
    print(m.dataset)
    for i in range(5):
        print("*******iteration number %d"%i)
        m.dataset = m.dataset.shuffle(seed = 3*i+1)
        m.train(epochs=40)
        #print("#################### test ################")
        #print("dataset test")
        #print(m.dataset["test"])
        test_pred = m.predict_by_dataset(m.dataset["test"])
        #print("test pred")
        #print(test_pred)
        true_labels = m.dataset["test"]["label"]
        print("true labels")
        print(true_labels)
        num_correct = np.sum([test_pred[i] == true_labels[i] for i in range(len(test_pred))])
        #print("num correct")
        #print(num_correct)
        print(num_correct/len(test_pred) * 100)
        M = multilabel_confusion_matrix(true_labels,test_pred,labels=list(range(2 if m.direction != None else 3)))
        print(M)
        #print(len(test_pred))