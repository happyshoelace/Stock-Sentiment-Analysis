import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_sentiment.py \"Your text here\"")
        return

    text = sys.argv[1]
    
    model_path = "./saved_sentiment_model"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Did you run train_model.py first to create the saved_sentiment_model?")
        return

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    
    # Inverse map of the labels used during training
    label_mapping = {0: "Bearish (Negative)", 1: "Neutral", 2: "Bullish (Positive)"}
    
    confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_id].item()
    
    print(f"\nText: {text}")
    print(f"Predicted Sentiment: {label_mapping[predicted_class_id]}")
    print(f"Confidence: {confidence:.2%}\n")

if __name__ == "__main__":
    main()
