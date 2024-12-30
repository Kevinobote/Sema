from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torch.nn as nn

class AudioTranscriber:
    def __init__(self, model_name="facebook/wav2vec2-large-xlsr-53", device="cpu"):
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
        
        # Optimize for CPU inference
        self.model.eval()
        if device == "cpu":
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
    
    def transcribe(self, audio_input):
        """Transcribe audio input to text."""
        with torch.no_grad():
            inputs = self.processor(
                audio_input, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            logits = self.model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)
            
        return transcription[0]