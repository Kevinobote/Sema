# Real-time Swahili Audio Processing System

## Setup Instructions

1. **Environment Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Running the Application**
   ```bash
   streamlit run src/main.py
   ```

## System Features

- Real-time audio transcription using Wav2Vec2
- Sentiment analysis with automatic Swahili-to-English translation
- Periodic summarization of transcribed content
- Optional sign language translation
- Dynamic visualization of transcriptions and sentiment
- Resource-efficient with CPU optimization

## Resource Management

The system implements several optimizations for CPU-only environments:

1. **Model Quantization**: All models are quantized for CPU inference
2. **Caching System**: Preprocessed audio is cached to avoid redundant processing
3. **Batch Processing**: Implemented where applicable for better CPU utilization
4. **Memory Management**: Automatic cleanup of old data and periodic summary generation

## Extending the System

To add support for new languages:

1. Update the `transcriber.py` with appropriate Wav2Vec2 model
2. Modify the translation pipeline in `sentiment_analyzer.py`
3. Update language-specific processing in `audio_processor.py`

## Troubleshooting

- **Memory Issues**: Adjust `history_size` in `visualizer.py`
- **CPU Performance**: Modify `chunk_size` in `audio_processor.py`
- **Cache Management**: Use `clear_cache()` in `data_loader.py`