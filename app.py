import streamlit as st
import time
import os
# Import the FakeNewsDetector class from model.py
from model import FakeNewsDetector
# Import utility functions for audio processing and text preprocessing from utils.py
from utils import audio_to_text, preprocess_text
import numpy as np # Although numpy is imported, it's not directly used in the final app.py, but kept for potential future use or if intermediate ML steps were added.

# --- Page Configuration ---
# Sets up the basic configuration for the Streamlit application page.
# page_title: Title displayed in the browser tab.
# page_icon: Favicon for the browser tab.
# layout: "wide" uses the full width of the browser, "centered" centers the content.
# initial_sidebar_state: "expanded" makes the sidebar visible by default.
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Glassmorphism and Modern UI ---
# Inject custom CSS to style the Streamlit application with a glassmorphism design.
# This includes custom fonts, background gradients, sidebar styling, and card layouts.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

/* Global font and text color settings */
html, body, [class*="st-"] {
    font-family: 'Poppins', sans-serif;
    color: #e0e0e0;
}

/* Body background with gradient */
body {
    background: linear-gradient(135deg, #1e0838 0%, #300a4b 100%);
    background-attachment: fixed;
}

/* App background with a subtle texture */
.stApp {
    background: url('https://www.transparenttextures.com/patterns/fabric-1.png') repeat;
}

/* Sidebar styling for a glassmorphism effect */
.stSidebar > div:first-child {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.18);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    border-radius: 0 10px 10px 0;
}

.sidebar .sidebar-content {
    background-color: rgba(255, 255, 255, 0.05);
}

/* Main content area styling for glassmorphism cards */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    padding: 20px;
    margin-bottom: 20px;
}

/* Header text color */
h1, h2, h3, h4, h5, h6 {
    color: #f0f0f0;
}

/* Button styling */
.stButton>button {
    background-color: #4CAF50; /* Green */
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
    transition: all 0.3s ease; /* Smooth transition for hover effects */
}
.stButton>button:hover {
    background-color: #45a049; /* Darker green on hover */
    transform: translateY(-2px); /* Slight lift effect */
    box-shadow: 0 4px 10px rgba(0,0,0,0.2); /* Shadow effect on hover */
}

/* Input field styling */
.stTextInput>div>div>input,
.stFileUploader>div>div>button {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.3);
    color: #f0f0f0;
    border-radius: 8px;
    padding: 10px;
}
.stTextInput>div>div>input:focus {
    border-color: #7952b3; /* Purple border on focus */
    box-shadow: 0 0 0 0.2rem rgba(121, 82, 179, 0.25);
}

/* Progress bar styling (for probability display) */
/* This targets the filled part of the progress bar */
.stProgress > div > div > div > div {
    background-color: #7952b3; /* Default purple color for progress bar */
}

/* Specific styling for the Streamlit progress bar track (background) */
/* This is a workaround to style the background of the progress bar for 'Fake' */
.st-emotion-cache-1pxazr7 {
    background: #FF4B4B; /* Red for Fake probability background */
}

/* Specific styling for the Streamlit progress bar fill (foreground) */
/* This sets the color for the 'Real' probability fill */
.st-emotion-cache-1pxazr7[data-testid="stProgressFilled"] {
    background: #5CB85C; /* Green for Real probability fill */
}

</style>
""", unsafe_allow_html=True)

# --- Initialize Detector (singleton pattern) ---
# Uses Streamlit's caching mechanism to load the FakeNewsDetector only once.
# This prevents reloading the model every time the script reruns, saving time and resources.
@st.cache_resource
def get_detector():
    """
    Initializes and returns a singleton instance of FakeNewsDetector.
    """
    return FakeNewsDetector()

detector = get_detector()

# --- Header and Sidebar ---
# Main title and description of the application.
st.title("AI Fake News Detector 🔍")
st.markdown("A tool to detect misinformation in text and video using ML.")

# Sidebar for navigation between different sections of the application.
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About", "Detect Text", "Detect Video"])

# --- Helper function for displaying results ---
def display_results(fake_probability: float, prediction_label: str):
    """
    Displays the analysis results in a stylized glassmorphism card.

    Args:
        fake_probability (float): The probability (0-100%) that the content is fake.
        prediction_label (str): The predicted label, either "FAKE" or "REAL".
    """
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Analysis Result")

    # Display predicted label with color coding (red for FAKE, green for REAL)
    st.write(f"**Predicted Label:** <span style='color: {'#FF4B4B' if prediction_label == 'FAKE' else '#5CB85C'}'>{prediction_label}</span>", unsafe_allow_html=True)
    
    # Display the probability of being fake
    st.write(f"**Probability of being FAKE:** {fake_probability:.2f}%")

    # Calculate real probability for the progress bar display
    real_probability = 100 - fake_probability
    # Display a progress bar showing the distribution between Real and Fake probabilities.
    st.progress(real_probability / 100, text=f"{real_probability:.2f}% Real vs {fake_probability:.2f}% Fake")

    # Provide a simple explanation based on the fake probability score.
    if fake_probability > 75:
        explanation = "This content exhibits strong indicators of misinformation, such as sensational claims, lack of credible sources, or emotional language."
    elif fake_probability > 50:
        explanation = "The content shows some characteristics often associated with fake news, but further investigation is recommended."
    elif fake_probability < 25:
        explanation = "This content appears to be credible, supported by factual statements and a balanced tone."
    else:
        explanation = "The content seems largely credible, but a slight degree of caution is advised."
    st.write(f"**Reason for classification:** {explanation}")

    # Provide a placeholder link for Google Fact Check.
    st.write("**Source-Check Links (Google Fact Check):**")
    st.markdown("- [Search on Google Fact Check](https://news.google.com/stories/)") # This link can be made dynamic with a proper API integration.
    st.markdown("</div>", unsafe_allow_html=True)


# --- Page Content Logic ---
# Renders content based on the selected page from the sidebar.
if page == "About":
    # About section: provides an overview of the project.
    st.header("About This Project")
    st.write("""
    Welcome to the AI Fake News Detector! This application leverages the power of 
    HuggingFace Transformers (specifically DistilBERT) to analyze news content 
    and determine its likelihood of being fake or real. 

    We support both text-based analysis (for articles, headlines, tweets) and 
    video analysis (by extracting audio and converting it to text).

    Our goal is to provide a quick and insightful initial assessment of information, 
    empowering users to be more critical consumers of news. 
    """)

    st.subheader("How it works:")
    st.markdown("""
    1. **Text Input**: You provide text (e.g., a news headline or article).
    2. **Video Input**: You upload a video file. Our system extracts the audio, 
       converts it into text using speech-to-text technology, and then processes 
       the extracted text.
    3. **AI Analysis**: The extracted or provided text is fed into a fine-tuned 
       DistilBERT model, which has been trained to identify patterns indicative 
       of fake or real news.
    4. **Result Display**: The application displays a probability score (e.g., 
       75% Fake), a summary explanation, and optionally, links to external fact-checking resources.
    """)
    st.info("Disclaimer: This tool is for informational purposes only and should not be considered a definitive source of truth. Always cross-reference with multiple credible sources.")

elif page == "Detect Text":
    # Detect Text section: allows users to input text for analysis.
    st.header("Detect Fake News from Text")
    text_input = st.text_area("Enter news text here (headline, article snippet, tweet)", height=200)
    if st.button("Analyze Text"):
        if text_input:
            with st.spinner("Analyzing text..."):
                # Preprocess the input text before feeding it to the model.
                processed_text = preprocess_text(text_input)
                # Get prediction from the FakeNewsDetector model.
                fake_probability, prediction_label = detector.predict(processed_text)
                # Display the results using the helper function.
                display_results(fake_probability, prediction_label)
        else:
            st.warning("Please enter some text to analyze.")

elif page == "Detect Video":
    # Detect Video section: allows users to upload video files for analysis.
    st.header("Detect Fake News from Video")
    video_file = st.file_uploader("Upload a video file (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

    if st.button("Analyze Video"):
        if video_file:
            with st.spinner("Processing video and analyzing content..."):
                # Save the uploaded video file temporarily to process.
                temp_video_path = os.path.join("temp_video.mp4")
                with open(temp_video_path, "wb") as f:
                    f.write(video_file.getbuffer())

                # Extract audio from the video and convert it to text.
                try:
                    extracted_text = audio_to_text(temp_video_path)
                    os.remove(temp_video_path) # Clean up the temporary video file after processing.

                    if extracted_text:
                        # Preprocess the extracted text and get prediction.
                        processed_text = preprocess_text(extracted_text)
                        fake_probability, prediction_label = detector.predict(processed_text)
                        # Display the results.
                        display_results(fake_probability, prediction_label)
                    else:
                        st.warning("Could not extract meaningful text from the video audio.")
                except Exception as e:
                    st.error(f"Error processing video: {e}")
                    # Ensure temporary file is removed even if an error occurs.
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
        else:
            st.warning("Please upload a video file to analyze.")