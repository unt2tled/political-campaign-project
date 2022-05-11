"""
This module contains CampaignTextModel class for training base/center model
"""
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
import numpy as np

BASE = "BASE"
CENTER = "CENTER"


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
        class_names = ["0", "1"]
        features = Features({'text': Value('string'), 'label': ClassLabel(names=class_names)})
        # Load dataset from csv
        self.dataset = load_dataset("csv", data_files=data_path, split="train", features=features)
        # Split data into train, test and validation
        train_testvalid = self.dataset.train_test_split(test_size=test_size)
        test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
        self.dataset = DatasetDict({
            "train": train_testvalid["train"],
            "test": test_valid["test"],
            "valid": test_valid["train"]})

    def train(self, learning_rate=2e-5, per_device_train_batch_size=16, per_device_eval_batch_size=16, epochs=5,
              decay=0.01):
        # Tokenizer tuning
        tokenizer = AutoTokenizer.from_pretrained(self.lang_model_name)
        tokenized_ds = self.dataset.map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(self.lang_model_name, num_labels=2)
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
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds["test"],
            tokenizer=tokenizer,
            compute_metrics=CampaignTextModel.compute_metrics,
            data_collator=data_collator,
        )
        # Train model
        trainer.train()
        self.trainer = trainer

    def predict_by_text(self, text_string: str, return_number=False):
        # Build dataset with one row
        data_to_predict = Dataset.from_dict({"text": text_string})
        predictions = self.trainer.predict(data_to_predict)
        pred_num = np.argmax(predictions.predictions, axis=-1)
        if return_number:
            return pred_num
        return self.direction if pred_num else "NOT_" + self.direction


if __name__ == "__main__":
    text = """
    The crime Medicare fraud. The victims American taxpayers. The boss Mitt Romney Romney supervised to company guilty. 
    A massive Medicare fraud. That's a fact. 25 million dollars in unnecessary blood tests. Right under Romney's nose. 
    """
    m = CampaignTextModel("distilbert-base-uncased", BASE)
    m.load_data_from_csv("tags_base.csv", 0.1)
    m.train(epochs=40)
    print(m.predict_by_text(text))
