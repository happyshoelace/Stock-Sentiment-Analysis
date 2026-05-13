import json
import csv
import torch
from transformers import pipeline
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def main():
    print("Loading stock mentions...")
    with open('Twitter Dataset\stockMentionsTWITTER.json', 'r', encoding='utf-8') as f:
        stock_mentions = json.load(f)
    
    # Flatten the dictionary into a list of (ticker, text)
    data = []
    for ticker, comments in stock_mentions.items():
        for comment in comments:
            data.append({"ticker": ticker, "text": comment})
    
    print(f"Total mentions found across all tickers: {len(data)}")
    
    # Initialize the zero-shot classification or sentiment analysis pipeline
    # ProsusAI/finbert is specifically fine-tuned for financial sentiment
    print("Loading FinBERT for auto-labeling...")
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
    
    labeled_data = []
    label_cache = {}  # Cache to avoid labeling the same text multiple times
    
    print("Auto-labeling data...")
    for item in tqdm(data, desc="Labeling"):
        text = item["text"]
        
        if text in label_cache:
            label = label_cache[text]
        else:
            try:
                # Let the pipeline handle truncation properly up to the max 512 tokens
                result = sentiment_pipeline(text, truncation=True, max_length=512)[0]
                label = result['label']
                label_cache[text] = label
            except Exception as e:
                print(f"Error processing text: {e}")
                continue
                
        labeled_data.append({
            "ticker": item["ticker"],
            "text": text,
            "label": label
        })

    # Convert to CSV and split
    print("Splitting and saving dataset...")
    # We only keep Positive and Negative for Bullish/Bearish, or keep Neutral as well.
    # We will map Positive -> 2, Neutral -> 1, Negative -> 0 later in the training script.
    
    train_data, val_data = train_test_split(labeled_data, test_size=0.2, random_state=42)
    
    def save_csv(filename, dataset):
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["ticker", "text", "label"])
            writer.writeheader()
            writer.writerows(dataset)
            
    save_csv('train_dataset.csv', train_data)
    save_csv('val_dataset.csv', val_data)
    print("Successfully created train_dataset.csv and val_dataset.csv")

if __name__ == "__main__":
    main()
