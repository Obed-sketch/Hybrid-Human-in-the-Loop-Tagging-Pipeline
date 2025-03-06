# STEP ONE: Data Collection & Preprocessing
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split

# Load raw data (columns: text, threat_type, severity)
data = pd.read_csv("security_incidents.csv")

# Clean text: remove special characters and lowercase
data["clean_text"] = data["text"].str.replace("[^a-zA-Z0-9 ]", "", regex=True).str.lower()

# Load spaCy for NLP preprocessing
nlp = spacy.load("en_core_web_sm")

# Extract entities (e.g., "malware", "phishing") as features
data["entities"] = data["clean_text"].apply(lambda x: [ent.text for ent in nlp(x).ents])

# Split into train/validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# STEP TWO Model Development (Auto-Tagging)
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Tokenize text
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(train_data["clean_text"].tolist(), truncation=True, padding=True)

# Convert tags to numerical labels
train_labels = train_data[["threat_type", "severity"]].apply(
    lambda x: f"{x['threat_type']}_{x['severity']}", axis=1
)
label_map = {label: idx for idx, label in enumerate(train_labels.unique())}
train_labels = train_labels.map(label_map).values

# Create PyTorch dataset
class SecurityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SecurityDataset(train_encodings, train_labels)

# Fine-tune BERT
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(label_map)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir="./logs",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()

# Save model
model.save_pretrained("auto_tagging_bert")

# STEP THREE Human Validation Interface
# Backend (Flask API):
# PYTHON
from flask import Flask, request, jsonify
import sqlite3

app = Flask(__name__)
conn = sqlite3.connect("tags.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS tags (id INTEGER PRIMARY KEY, text TEXT, auto_tag TEXT, validated_tag TEXT)")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]
    # Use BERT to generate auto-tag
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predicted_id = outputs.logits.argmax().item()
    auto_tag = list(label_map.keys())[predicted_id]
    return jsonify({"auto_tag": auto_tag})

@app.route("/validate", methods=["POST"])
def validate():
    text = request.json["text"]
    auto_tag = request.json["auto_tag"]
    validated_tag = request.json["validated_tag"]
    cursor.execute("INSERT INTO tags (text, auto_tag, validated_tag) VALUES (?, ?, ?)",
                  (text, auto_tag, validated_tag))
    conn.commit()
    return jsonify({"status": "success"})


  # Frontend (React Component):
  # JAVASCRIPT
  import React, { useState } from 'react';

function TaggingInterface() {
  const [text, setText] = useState("");
  const [autoTag, setAutoTag] = useState("");
  const [validatedTag, setValidatedTag] = useState("");

  const handlePredict = async () => {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const result = await response.json();
    setAutoTag(result.auto_tag);
  };

  const handleSubmit = async () => {
    await fetch("/validate", {
      method: "POST",
      body: JSON.stringify({ text, auto_tag: autoTag, validated_tag: validatedTag }),
    });
  };

  return (
    <div>
      <textarea onChange={(e) => setText(e.target.value)} />
      <button onClick={handlePredict}>Auto-Tag</button>
      <div>Auto Tag: {autoTag}</div>
      <input onChange={(e) => setValidatedTag(e.target.value)} placeholder="Correct tag..." />
      <button onClick={handleSubmit}>Submit</button>
    </div>
  );
}
  
# STEP FOUR Feedback Loop & Retraining
#PYTHON
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

def retrain_model():
    # Load new validated data from SQLite
    conn = sqlite3.connect("tags.db")
    new_data = pd.read_sql("SELECT text, validated_tag FROM tags", conn)
    new_data["threat_type"] = new_data["validated_tag"].str.split("_").str[0]
    new_data["severity"] = new_data["validated_tag"].str.split("_").str[1]
    
    # Append to training data and retrain BERT
    updated_data = pd.concat([data, new_data])
    # (Repeat Step 2's training process with updated_data)

dag = DAG(
    "retrain_pipeline",
    schedule_interval="@weekly",
    start_date=datetime(2023, 1, 1),
)

retrain_task = PythonOperator(
    task_id="retrain_model",
    python_callable=retrain_model,
    dag=dag,
)

# STEP FIVE Deployment & Monitoring
# Containerize the Flask API and React app:
# DOCKERFILE
# Flask backend
FROM python:3.8
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

# React frontend
FROM node:14
COPY frontend/ .
RUN npm install && npm run build
CMD ["npm", "start"]
