"""
This module contains methods for extracting text sentiment from texts
"""
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
# ref: https://colab.research.google.com/github/chrsiebert/sentiment-roberta-large-english/blob/main/sentiment_roberta_prediction_example.ipynb
# Create class for data preparation
class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}

class Sentiment_Extractor:
    def __init__(self,input_file_name,text_column,output_file_name):
        self.input_file_name = input_file_name
        self.text_column = text_column
        self.output_file_name = output_file_name
    def run(self):
        # Load tokenizer and model, create trainer
        model_name = "siebert/sentiment-roberta-large-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        trainer = Trainer(model=model)

        df_pred = pd.read_csv(self.input_file_name,encoding='cp1255')
        pred_texts = df_pred[self.text_column].dropna().astype('str').tolist()

        # Tokenize texts and create prediction data set
        tokenized_texts = tokenizer(pred_texts,truncation=True,padding=True)
        pred_dataset = SimpleDataset(tokenized_texts)

        # Run predictions
        predictions = trainer.predict(pred_dataset)

        # Transform predictions to labels
        preds = predictions.predictions.argmax(-1)
        labels = pd.Series(preds).map(model.config.id2label)
        scores = (np.exp(predictions[0])/np.exp(predictions[0]).sum(-1,keepdims=True)).max(1)

        # Create DataFrame with texts, predictions, labels, and scores
        df = pd.DataFrame(list(zip(pred_texts,preds,labels,scores)), columns=['text_sentiment','pred_sentiment','label_sentiment','score_sentiment'])
        df_output = df_pred.merge(df,left_on=self.text_column,right_on='text_sentiment')
        del df_output['text_sentiment']
        df_output.to_csv(self.output_file_name,encoding='cp1255',index=False)
        
if __name__ == "__main__":
    # Arguments
    # INPUT_FILE_NAME is the name of the input file
    INPUT_FILE_NAME = "tagging_MMD_db_with_summarized.csv"
    # TEXT_COLUMN is the name of the text column in the input file
    # from which we extract the positive / negative sentiment by the 🤗 model.
    TEXT_COLUMN = "text"
    OUTPUT_FILE_NAME = 'tagging_MMD_db_with_sentiment.csv'

    # Run Sentiment_Extractor on the given arguments
    obj = Sentiment_Extractor(INPUT_FILE_NAME,OUTPUT_FILE_NAME)
    obj.run()