import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

def main():
    json_file = 'stockMentions.json'
    model_path = "./saved_sentiment_model"
    
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found.")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run train_model.py first.")
        return

    # 1. Load Model
    print(f"Loading trained model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 2. Load Data
    with open(json_file, 'r', encoding='utf-8') as f:
        stock_mentions = json.load(f)
    
    label_mapping = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
    
    results = {}

    # 3. Process each ticker
    print("\nProcessing mentions and predicting sentiment...")
    for ticker, comments in stock_mentions.items():
        ticker_summary = {"Bearish": 0, "Neutral": 0, "Bullish": 0, "mentions": []}
        
        for comment in comments:
            # Tokenize and predict
            inputs = tokenizer(comment, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            pred_id = outputs.logits.argmax().item()
            sentiment = label_mapping[pred_id]
            
            ticker_summary[sentiment] += 1
            # Only keep a snippet for the final report if it's not neutral
            if sentiment != "Neutral":
                ticker_summary["mentions"].append((comment[:100], sentiment))
        
        results[ticker] = ticker_summary

    # 4. Display Summary Report
    print("\n" + "="*60)
    print(f"{'TICKER':<10} | {'BULLISH':<10} | {'NEUTRAL':<10} | {'BEARISH':<10}")
    print("-" * 60)
    
    for ticker, summary in results.items():
        print(f"{ticker:<10} | {summary['Bullish']:<10} | {summary['Neutral']:<10} | {summary['Bearish']:<10}")

    print("\n" + "="*60)
    print("KEY SENTIMENT FINDINGS")
    print("="*60)
    
    found_interesting = False
    for ticker, summary in results.items():
        if summary["Bullish"] > 0 or summary["Bearish"] > 0:
            found_interesting = True
            print(f"\n[{ticker}]")
            for snippet, sent in summary["mentions"][:2]: # Show up to 2 snippets
                icon = "[+]" if sent == "Bullish" else "[-]"
                # Also strip any problematic characters from snippet just in case
                safe_snippet = snippet.encode('ascii', 'ignore').decode('ascii')
                print(f"  {icon} {sent}: {safe_snippet}...")
    
    if not found_interesting:
        print("No strong bullish or bearish sentiment found. Most mentions were neutral.")

if __name__ == "__main__":
    main()
