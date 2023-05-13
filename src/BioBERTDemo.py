import torch
from torch.cuda import device
from transformers import AutoTokenizer, AutoModel

model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, load_metric
from seqeval.metrics import f1_score

# Load dataset and metric
dataset = load_dataset("text", data_files=os.path.join("NCBI-disease", "NCBI_corpus_testing.txt"))
metric = load_metric("seqeval")

# Load tokenizer and model
model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=3)

# Tokenize input data
def tokenize(example):
    tokenized = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    labels = example["label"]
    tokenized["label"] = [-100 if label == "O" else model.config.id2label.index(label) for label in labels]
    return tokenized

tokenized_dataset = dataset.map(tokenize, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
)

# Define trainer and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=lambda data: {"input_ids": torch.stack([item["input_ids"] for item in data]),
                                "attention_mask": torch.stack([item["attention_mask"] for item in data]),
                                "labels": torch.stack([item["label"] for item in data])},
    compute_metrics=lambda p: {"f1": f1_score(p["predictions"], p["label_ids"])}
)

trainer.train()

# Evaluate the trained model
predictions, labels = [], []
for example in tokenized_dataset["test"]:
    input_ids = example["input_ids"].unsqueeze(0)
    attention_mask = example["attention_mask"].unsqueeze(0)
    outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
    predictions.append(outputs.logits.argmax(-1).squeeze().tolist())
    labels.append(example["ner_tags"])

# Flatten the predictions and labels to compute the metrics
predictions = [tag for sent in predictions for tag in sent]
labels = [tag for sent in labels for tag in sent]

# Compute the F1 score using seqeval
f1 = f1_score(labels, predictions)

# Print the F1 score
print("F1 score: {:.2f}".format(f1 * 100))
