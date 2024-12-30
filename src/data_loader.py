from datasets import load_dataset
import torch
import torchaudio
import os
import json
from pathlib import Path
import numpy as np

class DataLoader:
    def __init__(self, cache_dir="./cache", sample_rate=16000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.sample_rate = sample_rate
        self.dataset = None
        self.cache_index = {}
        self._load_cache_index()

    def _load_cache_index(self):
        """Load cache index from disk."""
        index_path = self.cache_dir / "cache_index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                self.cache_index = json.load(f)

    def _save_cache_index(self):
        """Save cache index to disk."""
        with open(self.cache_dir / "cache_index.json", "w") as f:
            json.dump(self.cache_index, f)

    def load_mozilla_common_voice(self, split="train"):
        """Load Mozilla Common Voice dataset for Swahili."""
        self.dataset = load_dataset(
            "mozilla-foundation/common_voice_11_0",
            "sw",
            split=split,
            trust_remote_code=True
        )
        return self.dataset

    def preprocess_and_cache(self, batch_size=32):
        """Preprocess audio files and cache them."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_mozilla_common_voice first.")

        for i in range(0, len(self.dataset), batch_size):
            batch = self.dataset[i:i + batch_size]
            
            for idx, item in enumerate(batch):
                cache_key = f"{item['path']}_{self.sample_rate}"
                cache_path = self.cache_dir / f"{cache_key}.pt"
                
                if cache_key not in self.cache_index:
                    # Load and preprocess audio
                    waveform, sr = torchaudio.load(item["path"])
                    if sr != self.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                        waveform = resampler(waveform)
                    
                    # Normalize
                    waveform = (waveform - waveform.mean()) / torch.std(waveform)
                    
                    # Cache processed audio
                    torch.save(waveform, cache_path)
                    self.cache_index[cache_key] = str(cache_path)
            
            # Save cache index periodically
            if i % (batch_size * 10) == 0:
                self._save_cache_index()
        
        # Final save of cache index
        self._save_cache_index()

    def get_cached_audio(self, audio_path):
        """Retrieve preprocessed audio from cache."""
        cache_key = f"{audio_path}_{self.sample_rate}"
        if cache_key in self.cache_index:
            return torch.load(self.cache_index[cache_key])
        return None

    def clear_cache(self):
        """Clear the cache directory."""
        for file_path in self.cache_dir.glob("*.pt"):
            file_path.unlink()
        self.cache_index = {}
        self._save_cache_index()