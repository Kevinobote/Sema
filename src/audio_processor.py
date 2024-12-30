import torch
import torchaudio
import librosa
import numpy as np
import sounddevice as sd

class AudioProcessor:
    def __init__(self, sample_rate=16000, chunk_size=4096):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
    def load_audio(self, file_path):
        """Load and preprocess audio file."""
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        return waveform
    
    def preprocess_audio(self, waveform):
        """Preprocess audio for model input."""
        # Ensure mono audio
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Normalize
        waveform = (waveform - waveform.mean()) / torch.std(waveform)
        return waveform
    
    def start_stream(self, callback):
        """Start real-time audio streaming."""
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")
            audio_data = torch.from_numpy(indata[:, 0]).unsqueeze(0)
            callback(self.preprocess_audio(audio_data))
        
        stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            callback=audio_callback
        )
        return stream