"""
This module contains loaders for loading models to predict a political campaign orientation (base/center)
"""
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, Trainer
from datasets import load_metric
import pandas as pd
import numpy as np
import torch
from torch.nn.functional import softmax

HF_TOKEN = "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"

class HFPretrainedModel:
    def __init__(self, lang_model_name: str,checkpoint:str):
        self.lang_model_name = lang_model_name
        self.checkpoint = checkpoint
        self.init_tokenizer()
        self.init_config()
    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        metric = load_metric("accuracy")
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    def init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.lang_model_name)
    def init_config(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, use_auth_token=HF_TOKEN, num_labels=2)
        self.trainer = Trainer(model=self.model,tokenizer=self.tokenizer,compute_metrics=HFPretrainedModel.compute_metrics)
    def predict(self, data: dict):
        # Build dataset with one row
        data_to_predict = Dataset.from_dict(data)
        tokenized_ds = data_to_predict.map(lambda examples: self.tokenizer([examples[text_feature] if examples[text_feature] is not None else '' for text_feature in data.keys()],is_split_into_words=True,truncation=True))
        predictions = self.trainer.predict(tokenized_ds)
        pred_tensor = torch.tensor(predictions.predictions[0])
        return softmax(pred_tensor, dim=0)
