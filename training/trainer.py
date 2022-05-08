from transformers import AutoTokenizer
from datasets import load_dataset
from datasets import DatasetDict
from datasets import Features
from datasets import Value
from datasets import ClassLabel
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

LANG_MODEL = "distilbert-base-uncased"

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

class_names = ["0", "1"]
features = Features({'text': Value('string'), 'label': ClassLabel(names=class_names)})

dataset = load_dataset("csv", data_files="tags_base.csv", split="train", features=features)
# Split data into train, test and validation
train_testvalid = dataset.train_test_split(test_size=0.1)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})

tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL)
tokenized_ds = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(LANG_MODEL, num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
predictions = trainer.predict(tokenized_ds["valid"])
print("Here: "+str(predictions.label_ids))
