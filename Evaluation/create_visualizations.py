import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm
import os

def main():
    # 1. Setup
    json_file = 'Twitter Dataset\stockMentionsTWITTER.json'
    model_path = "Approach 1 - Model Training/saved_sentiment_model"
    output_dir = "Analysis Results/plots"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(json_file) or not os.path.exists(model_path):
        print("Required files missing.")
        return

    # 2. Data & Labels
    with open(json_file, 'r', encoding='utf-8') as f:
        stock_mentions = json.load(f)
    
    data = []
    for ticker, comments in stock_mentions.items():
        for comment in comments:
            data.append(comment)

    print("Generating labels for visualization...")
    device = 0 if torch.cuda.is_available() else -1
    teacher_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    student_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model.to(current_device)
    student_model.eval()

    true_labels = []
    pred_labels = []
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    teacher_map = {"negative": 0, "neutral": 1, "positive": 2}

    for text in tqdm(data, desc="Processing"):
        # Teacher
        try:
            t_res = teacher_pipeline(text[:2000], truncation=True, max_length=512)[0]
            true_labels.append(t_res['label'].lower().strip())
        except:
            true_labels.append("neutral")
        
        # Student
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(current_device)
        with torch.no_grad():
            s_outputs = student_model(**inputs)
        pred_id = s_outputs.logits.argmax().item()
        pred_labels.append(label_map[pred_id])

    # 3. Plot Confusion Matrix
    labels = ["negative", "neutral", "positive"]
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix: Teacher vs Student Model')
    plt.ylabel('Teacher (Ground Truth)')
    plt.xlabel('Student (Trained Model)')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # 4. Plot Sentiment Distribution
    df = pd.DataFrame({
        'Model': ['Teacher']*len(true_labels) + ['Student']*len(pred_labels),
        'Sentiment': true_labels + pred_labels
    })
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Sentiment', hue='Model', order=labels)
    plt.title('Sentiment Distribution Comparison')
    plt.ylabel('Count')
    dist_path = os.path.join(output_dir, 'sentiment_distribution.png')
    plt.savefig(dist_path)
    print(f"Distribution plot saved to {dist_path}")

    print("\nVisualizations complete!")

if __name__ == "__main__":
    main()
