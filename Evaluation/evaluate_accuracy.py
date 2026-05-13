import json
import torch
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import os

def main():
    # 1. Load data
    json_file = 'Twitter Dataset\stockMentionsTWITTER.json'
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found.")
        return

    print(f"Loading {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        stock_mentions = json.load(f)
    
    data = []
    for ticker, comments in stock_mentions.items():
        for comment in comments:
            data.append(comment)
    
    print(f"Total mentions to test: {len(data)}")

    # 2. Get Ground Truth from FinBERT (Teacher)
    print("Loading FinBERT for ground truth labeling...")
    device = 0 if torch.cuda.is_available() else -1
    # Use the same model as in create_dataset.py
    teacher_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
    
    true_labels = []
    print("Generating ground truth labels (this may take a minute)...")
    # Batch processing would be faster, but let's stick to the current pattern
    for text in tqdm(data, desc="Teacher labeling"):
        try:
            # Truncate to 512 as FinBERT/BERT max length
            res = teacher_pipeline(text[:2000], truncation=True, max_length=512)[0]
            label = res['label'].lower().strip()
            true_labels.append(label)
        except Exception as e:
            # print(f"Error labeling: {e}")
            true_labels.append("neutral") # Fallback

    # 3. Get Predictions from our Trained Model (Student)
    model_path = "./saved_sentiment_model"
    if not os.path.exists(model_path):
        print(f"Error: Trained model directory {model_path} not found. Please run train_model.py first.")
        return

    print(f"Loading student model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    student_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Move model to device
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model.to(current_device)
    student_model.eval()
    
    student_preds = []
    # Match the mapping in train_model.py: {"negative": 0, "neutral": 1, "positive": 2}
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    
    print("Generating student predictions...")
    for text in tqdm(data, desc="Student predicting"):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(current_device)
        with torch.no_grad():
            outputs = student_model(**inputs)
        pred_id = outputs.logits.argmax().item()
        student_preds.append(label_map[pred_id])

    # 4. Calculate and Display Results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    acc = accuracy_score(true_labels, student_preds)
    print(f"Overall Model Accuracy: {acc:.2%}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, student_preds, target_names=["negative", "neutral", "positive"]))
    
    # Sample disagreements
    disagreements = []
    for i in range(len(data)):
        if true_labels[i] != student_preds[i]:
            disagreements.append((data[i], true_labels[i], student_preds[i]))
    
    if disagreements:
        print("\nSample Disagreements (Teacher vs Student):")
        for i in range(min(3, len(disagreements))):
            text, true, pred = disagreements[i]
            print(f"- Text: {text[:100]}...")
            print(f"  Teacher: {true}")
            print(f"  Student: {pred}\n")

if __name__ == "__main__":
    main()
