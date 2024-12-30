import streamlit as st
import plotly.graph_objects as go
from collections import deque
import time

class RealTimeVisualizer:
    def __init__(self, history_size=100):
        self.history_size = history_size
        self.transcription_history = deque(maxlen=history_size)
        self.sentiment_history = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        self.summary = ""
        self.sign_language_enabled = False
        
    def setup_streamlit(self):
        """Initialize Streamlit dashboard."""
        st.title("Real-time Swahili Audio Processing")
        
        # Add sign language toggle
        self.sign_language_enabled = st.sidebar.checkbox("Enable Sign Language Translation")
        
        # Create containers for different components
        self.text_container = st.container()
        with self.text_container:
            self.text_placeholder = st.empty()
            self.summary_placeholder = st.empty()
        
        self.visual_container = st.container()
        with self.visual_container:
            self.sentiment_plot = st.empty()
            if self.sign_language_enabled:
                self.sign_placeholder = st.empty()
        
    def update_display(self, transcription, sentiment, summary=None, sign_translation=None):
        """Update the real-time display with new data."""
        current_time = time.time()
        
        # Update histories
        self.transcription_history.append(transcription)
        self.sentiment_history.append(sentiment['score'])
        self.timestamps.append(current_time)
        
        # Update transcription display
        self.text_placeholder.text_area(
            "Latest Transcription",
            transcription,
            height=100
        )
        
        # Update summary if provided
        if summary:
            self.summary = summary
            self.summary_placeholder.text_area(
                "Summary",
                self.summary,
                height=150
            )
        
        # Update sign language translation if enabled and provided
        if self.sign_language_enabled and sign_translation:
            self.sign_placeholder.text_area(
                "Sign Language Translation",
                sign_translation,
                height=100
            )
        
        # Update sentiment plot
        fig = go.Figure(data=go.Scatter(
            x=list(self.timestamps),
            y=list(self.sentiment_history),
            mode='lines+markers'
        ))
        fig.update_layout(
            title="Sentiment Analysis Over Time",
            xaxis_title="Time",
            yaxis_title="Sentiment Score",
            height=400
        )
        self.sentiment_plot.plotly_chart(fig)