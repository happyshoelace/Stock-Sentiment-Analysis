import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tqdm import tqdm
import os

def main():
    val_file = 'val_dataset.csv'
    model_path = "./saved_sentiment_model"
    output_dir = "./results/plots_twitter"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(val_file):
        print(f"Error: {val_file} not found.")
        return

    print("Loading validation dataset...")
    df = pd.read_csv(val_file)
    # Ensure text and label columns are clean
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    
    print(f"Loading trained model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    texts = df['text'].tolist()
    true_labels = df['label'].tolist()
    pred_labels = []
    label_map = {0: "negative", 1: "neutral", 2: "positive"}

    print(f"Predicting sentiment for {len(texts)} mentions (using batching)...")
    batch_size = 64 # Use a larger batch size for speed
    for i in tqdm(range(0, len(texts), batch_size), desc="Batch predicting"):
        batch_texts = texts[i:i+batch_size]
        # Handle cases where batch_texts might be empty or invalid
        try:
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            pred_labels.extend([label_map[p] for p in preds])
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # Fallback for the batch
            pred_labels.extend(["neutral"] * len(batch_texts))

    # 3. Plot Confusion Matrix
    labels = ["negative", "neutral", "positive"]
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix: Twitter Validation Set')
    plt.ylabel('FinBERT (Teacher)')
    plt.xlabel('DistilBERT (Student)')
    cm_path = os.path.join(output_dir, 'twitter_confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # 4. Plot Sentiment Distribution
    plot_df = pd.DataFrame({
        'Model': ['Teacher']*len(true_labels) + ['Student']*len(pred_labels),
        'Sentiment': true_labels + pred_labels
    })
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=plot_df, x='Sentiment', hue='Model', order=labels)
    plt.title('Twitter Sentiment Distribution Comparison')
    plt.ylabel('Count')
    dist_path = os.path.join(output_dir, 'twitter_sentiment_distribution.png')
    plt.savefig(dist_path)
    print(f"Distribution plot saved to {dist_path}")

    print("\n" + "="*50)
    print("TWITTER VALIDATION RESULTS")
    print("="*50)
    acc = accuracy_score(true_labels, pred_labels)
    print(f"Overall Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=labels))

if __name__ == "__main__":
    main()
