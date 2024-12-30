import torch
from audio_processor import AudioProcessor
from transcriber import AudioTranscriber
from sentiment_analyzer import SentimentAnalyzer
from visualizer import RealTimeVisualizer
from data_loader import DataLoader
from summarizer import TextSummarizer
from sign_language import SignLanguageTranslator
import streamlit as st

def main():
    try:
        # Initialize components
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.sidebar.info(f"Running on {device}")
        
        # Initialize data loader and load dataset
        with st.spinner("Loading and preprocessing dataset..."):
            data_loader = DataLoader(cache_dir="./cache")
            dataset = data_loader.load_mozilla_common_voice()
            data_loader.preprocess_and_cache()
        
        audio_processor = AudioProcessor()
        transcriber = AudioTranscriber(device=device)
        sentiment_analyzer = SentimentAnalyzer(device=device)
        summarizer = TextSummarizer(device=device)
        sign_translator = SignLanguageTranslator(device=device)
        visualizer = RealTimeVisualizer()
        
        # Setup visualization
        visualizer.setup_streamlit()
        
        # Initialize state variables
        recent_transcriptions = []
        last_summary_time = time.time()
        summary_interval = 60  # Generate summary every 60 seconds
        
        # Processing callback
        def process_audio(audio_data):
            try:
                # Transcribe audio
                transcription = transcriber.transcribe(audio_data)
                recent_transcriptions.append(transcription)
                
                # Analyze sentiment
                sentiment = sentiment_analyzer.analyze(transcription)
                
                # Generate summary periodically
                current_time = time.time()
                summary = None
                if current_time - last_summary_time >= summary_interval and recent_transcriptions:
                    summary = summarizer.summarize(recent_transcriptions)
                    last_summary_time = current_time
                    recent_transcriptions.clear()
                
                # Generate sign language translation if enabled
                sign_translation = None
                if visualizer.sign_language_enabled:
                    sign_translation = sign_translator.text_to_signs(transcription)
                
                # Update visualization
                visualizer.update_display(
                    transcription=transcription,
                    sentiment=sentiment,
                    summary=summary,
                    sign_translation=sign_translation
                )
                
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
                time.sleep(1)  # Prevent error spam
        
        # Start audio stream
        with audio_processor.start_stream(process_audio) as stream:
            try:
                while True:
                    stream.sleep(0.1)
            except KeyboardInterrupt:
                pass
            except Exception as e:
                st.error(f"Stream error: {str(e)}")
            finally:
                # Cleanup
                sign_translator.cleanup()
                if len(recent_transcriptions) > 0:
                    final_summary = summarizer.summarize(recent_transcriptions)
                    visualizer.update_display(
                        transcription="Session ended.",
                        sentiment={'score': 0, 'label': 'NEUTRAL'},
                        summary=final_summary
                    )
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()