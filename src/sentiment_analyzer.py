from transformers import pipeline
import torch

class SentimentAnalyzer:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-sw-en", device="cpu"):
        self.device = device
        self.translator = pipeline(
            "translation",
            model=model_name,
            device=0 if device == "cuda" else -1
        )
        
        # Initialize sentiment analysis for English text
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if device == "cuda" else -1
        )
        
    def analyze(self, text):
        """Analyze sentiment of Swahili text by first translating to English."""
        # Translate Swahili to English
        english_text = self.translator(text)[0]['translation_text']
        
        # Analyze sentiment
        sentiment = self.sentiment_analyzer(english_text)[0]
        return {
            'label': sentiment['label'],
            'score': sentiment['score'],
            'english_text': english_text
        }