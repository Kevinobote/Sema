# **Real-time Audio-to-Text Multilingual Sentiment Analysis**

## **Overview**  
This project aims to build a comprehensive system capable of transcribing audio into text in multiple languages and conducting sentiment analysis on the transcribed text. By leveraging the Mozilla Dataset Corpus, which contains over 46,000 Swahili audio files and corresponding transcriptions, the system will provide actionable insights through sentiment analysis, visualization, and reporting. The system also incorporates real-world testing with audio extracted from YouTube videos to ensure its robustness and applicability in diverse scenarios.  

---

## **Features**  
- Real-time transcription of audio files to text, supporting multiple languages.  
- Sentiment analysis to identify polarity (positive, negative, neutral) from transcribed text.  
- Dynamic visualizations, including sentiment timelines and word clouds, for effective communication of insights.  
- Ability to process audio from diverse sources, including YouTube.  

---

## **Current Progress**  

### **Step 1: Exploratory Data Analysis (EDA)**  
- Conducted detailed analysis of the Mozilla Dataset Corpus to understand dataset characteristics.  
- Verified the alignment between audio files and their corresponding text transcriptions to ensure data quality.  

### **Step 2: Audio Processing with YouTube Data**  
- Successfully extracted audio from YouTube videos for testing the transcription process.  
- Transcribed the audio into text and performed sentiment analysis to evaluate the system’s accuracy.  
- Generated preliminary visualizations, such as word clouds, to summarize the extracted insights.  

---

## **End Goal**  
The final system will:  
1. Provide seamless transcription and sentiment analysis for audio in multiple languages.  
2. Support various real-world audio sources, including user-uploaded files and live streams.  
3. Deliver actionable insights through interactive visualizations and detailed reporting.  

---

## **Next Steps**  

### **1. Enhancing Audio Feature Extraction**  
- Introduce advanced features like Chroma, Spectral Roll-off, and Zero Crossing Rate for richer audio analysis.  
- Implement Voice Activity Detection (VAD) to filter out non-speech segments for improved focus.  

### **2. Refining Text Processing**  
- Perform advanced text preprocessing to remove extraneous elements for cleaner analysis.  
- Use Named Entity Recognition (NER) and Part-of-Speech Tagging to extract and analyze key entities and text characteristics.  

### **3. Advanced Sentiment Analysis**  
- Integrate models like VADER or transformer-based solutions (e.g., BERT) for enhanced sentiment detection.  
- Expand sentiment analysis to include emotion detection and aspect-based analysis for nuanced insights.  

### **4. Visualization Enhancements**  
- Develop interactive sentiment timelines and keyword-based word clouds.  
- Enable speaker-specific sentiment analysis through speaker diarization techniques.  

### **5. Real-Time and Scalable Processing**  
- Enable real-time transcription and analysis for live audio streams.  
- Automate workflows triggered by specific sentiment detections, such as alerts for negative sentiment.  

---

## **How to Use**  

1. **Setup**:  
   Clone this repository and install the required dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

2. **Process Audio**:  
   To analyze an audio file:  
   ```bash  
   python main.py --audio_file sample_audio.wav  
   ```  

3. **Output**:  
   - Transcription of the input audio.  
   - Sentiment analysis results (positive, negative, neutral).  
   - Visualizations like word clouds and sentiment graphs.  


## **Future Directions**  
- Expand the system’s capabilities to support more languages and accents.  
- Develop a web and mobile interface for user-friendly interaction.  
- Automate reporting for comprehensive transcription and sentiment summaries.  


## **Contributing**  
We welcome contributions! Fork the repository, create your feature branch, and submit a pull request for review.  


## **License**  
This project is licensed under the MIT License.  



