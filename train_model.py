import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    print("Loading datasets...")
    train_df = pd.read_csv('train_dataset.csv')
    val_df = pd.read_csv('val_dataset.csv')
    
    # Map labels to integers
    # Clean labels to ensure matching (strip spaces and lowercase)
    train_df['label'] = train_df['label'].astype(str).str.strip().str.lower()
    val_df['label'] = val_df['label'].astype(str).str.strip().str.lower()
    
    label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    train_df['label'] = train_df['label'].map(label_mapping)
    val_df['label'] = val_df['label'].map(label_mapping)
    
    # Drop NaNs just in case
    train_df = train_df.dropna(subset=['text', 'label'])
    val_df = val_df.dropna(subset=['text', 'label'])
    
    # CLASS BALANCING: The dataset is 93% Neutral.
    # We will downsample the Neutral class to match the size of the smaller classes
    # so the model actually learns Bullish/Bearish features.
    def balance_data(df):
        df_pos = df[df['label'] == 2]
        df_neg = df[df['label'] == 0]
        df_neu = df[df['label'] == 1]
        
        # We'll take all pos/neg and a limited amount of neutral
        n_samples = max(len(df_pos), len(df_neg)) * 2 # Keep neutral at 2x the size of others
        df_neu_downsampled = df_neu.sample(n=min(len(df_neu), n_samples), random_state=42)
        
        return pd.concat([df_pos, df_neg, df_neu_downsampled]).sample(frac=1, random_state=42)

    print(f"Original training size: {len(train_df)}")
    train_df = balance_data(train_df)
    print(f"Balanced training size: {len(train_df)}")
    print(f"New label distribution:\n{train_df['label'].value_counts()}")

    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    print("Loading DistilBERT tokenizer...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing data...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

    print("Saving model...")
    model.save_pretrained("./saved_sentiment_model")
    tokenizer.save_pretrained("./saved_sentiment_model")
    print("Model saved to ./saved_sentiment_model")

if __name__ == "__main__":
    main()
