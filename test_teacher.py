from transformers import pipeline
import torch

def main():
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
    
    texts = [
        "Bullish on NVIDIA, the new Blackwell chips are going to drive revenue to the moon next quarter!",
        "I'm worried about the high interest rates and the geopolitical risks in the Middle East hitting tech stocks hard."
    ]
    
    for text in texts:
        result = sentiment_pipeline(text)[0]
        print(f"Text: {text}")
        print(f"Label: {result['label']}")
        print(f"Score: {result['score']}\n")

if __name__ == "__main__":
    main()
