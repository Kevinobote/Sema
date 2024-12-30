from transformers import pipeline
import torch
from typing import List

class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn", device="cpu"):
        self.device = device
        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            device=0 if device == "cuda" else -1
        )
        
    def summarize(self, texts: List[str], max_length: int = 130, min_length: int = 30):
        """Generate summary from a list of transcribed texts."""
        # Combine texts with proper spacing
        combined_text = " ".join(texts)
        
        # Generate summary
        summary = self.summarizer(
            combined_text, 
            max_length=max_length, 
            min_length=min_length,
            do_sample=False
        )[0]['summary_text']
        
        return summary