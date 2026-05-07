import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
import sys

def main():
    # --- CONFIGURATION ---
    if len(sys.argv) < 2:
        print("Usage: python analyze_stocks.py <json_file>")
        print("Example: python analyze_stocks.py stockMentions.json")
        return

    input_file = sys.argv[1]
    model_path = "./saved_sentiment_model"
    output_dir = "./analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run train_model.py first.")
        return

    # --- 1. LOAD MODEL ---
    print(f"Loading trained model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # --- 2. LOAD DATA ---
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        stock_mentions = json.load(f)
    
    # Flatten data for batching
    all_mentions = []
    for ticker, comments in stock_mentions.items():
        for comment in comments:
            all_mentions.append({"ticker": ticker, "text": comment})
    
    print(f"Total mentions to process: {len(all_mentions)}")

    # --- 3. BATCH PREDICTION ---
    label_map = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
    results_list = []
    batch_size = 64
    
    print("Predicting sentiment...")
    for i in tqdm(range(0, len(all_mentions), batch_size), desc="Analyzing"):
        batch = all_mentions[i : i + batch_size]
        texts = [m["text"] for m in batch]
        
        try:
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            
            for j, pred in enumerate(preds):
                results_list.append({
                    "ticker": batch[j]["ticker"],
                    "text": batch[j]["text"],
                    "sentiment": label_map[pred]
                })
        except Exception as e:
            print(f"Error in batch: {e}")
            continue

    # --- 4. AGGREGATE RESULTS ---
    df = pd.DataFrame(results_list)
    ticker_summary = df.groupby('ticker')['sentiment'].value_counts().unstack(fill_value=0)
    
    # Ensure all columns exist
    for col in ["Bullish", "Neutral", "Bearish"]:
        if col not in ticker_summary.columns:
            ticker_summary[col] = 0
            
    ticker_summary['Total'] = ticker_summary["Bullish"] + ticker_summary["Neutral"] + ticker_summary["Bearish"]
    
    # --- 5. CALCULATE SENTIMENT SCORE (VIBE SCORE) ---
    # Score formula: (Bullish - Bearish) / Total 
    # Range: -1.0 (Totally Bearish) to +1.0 (Totally Bullish)
    ticker_summary['Vibe Score'] = (ticker_summary['Bullish'] - ticker_summary['Bearish']) / ticker_summary['Total']
    
    # --- 6. TOP STOCKS REPORT ---
    print("\n" + "="*60)
    print(f"{'TICKER':<10} | {'BULLISH':<10} | {'BEARISH':<10} | {'VIBE SCORE':<12}")
    print("-" * 60)
    
    # Show top 10 tickers by total mentions, but sorted by score
    top_movers = ticker_summary.sort_values(by="Vibe Score", ascending=False)
    # Filter for tickers with at least 1 non-neutral mention to make it interesting
    interesting_movers = top_movers[(top_movers['Bullish'] > 0) | (top_movers['Bearish'] > 0)]
    
    for ticker, row in interesting_movers.head(10).iterrows():
        score = row['Vibe Score']
        indicator = "(+)" if score > 0 else "(-)"
        print(f"{ticker:<10} | {row['Bullish']:<10} | {row['Bearish']:<10} | {score:>10.2f} {indicator}")

    # --- 7. VISUALIZATIONS ---
    print("\nGenerating improved plots...")
    
    # Plot 1: Overall Sentiment Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='sentiment', order=["Bearish", "Neutral", "Bullish"], palette="RdYlGn")
    plt.title(f'Overall Sentiment Distribution - {input_file}')
    plt.savefig(os.path.join(output_dir, "sentiment_distribution.png"))
    
    # Plot 2: Top 15 Stocks by Vibe Score
    plt.figure(figsize=(12, 8))
    top_15_scores = interesting_movers.head(15).sort_values(by='Vibe Score')
    colors = ['green' if x > 0 else 'red' for x in top_15_scores['Vibe Score']]
    top_15_scores['Vibe Score'].plot(kind='barh', color=colors)
    plt.title('Top 15 Stocks by Sentiment "Vibe Score" (Bullish vs Bearish)')
    plt.xlabel('Sentiment Score (-1 to +1)')
    plt.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_vibe_scores.png"))

if __name__ == "__main__":
    main()
