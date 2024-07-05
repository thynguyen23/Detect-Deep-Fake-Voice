import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import base64

# Load the pre-trained LSTM model
model = tf.keras.models.load_model('cnn_lstm_fold_5.h5')

# Function to extract features from audio file
def extract_features(file_path, target_duration=60.0, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    max_len = int(target_duration * sr)  # Target length in samples

    if len(y) < max_len:
        # Pad the signal to the target length
        y = np.pad(y, (0, max_len - len(y)), mode='constant')
    else:
        # Truncate the signal to the target length
        y = y[:max_len]
    
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr).mean()
    rms = librosa.feature.rms(y=y).mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs = [mfcc[i].mean() for i in range(20)]
    
    features = np.array([
        chroma_stft,
        rms,
        spectral_centroid,
        spectral_bandwidth,
        rolloff,
        zero_crossing_rate,
        *mfccs
    ])
    
    return features

# Function to encode image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Streamlit application code
st.set_page_config(page_title="Deepfake Voice Detection", page_icon="🎤", layout="wide")

# CSS styling
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #1B263B;
        color: #EAEAEA;
    }
    .stApp {
        max-width: 650px;
        margin: 0 auto;
        padding-top: 10px; /* Add padding to the top to avoid unnecessary white space */
    }
    .logo {
        position: fixed;
        bottom: 30px;
        right: 10px;
        animation: fadeIn 2s ease-in-out;
    }
    .header {
        text-align: center;
        font-size: 35px;
        color: #00A8E8;
        margin-top: 20px;
        animation: slideIn 1s ease-out;
    }
    .info {
        text-align: center;
        font-size: 16px;
        color: #BDC3C7;
        margin-bottom: 20px;
    }
    .file-uploader {
        margin: 0 auto;
        max-width: 600px;
        text-align: center;
    }
    .file-uploader label {
        display: inline-block;
        width: auto;
        padding: 10px;
        border: 2px dashed #00A8E8;
        border-radius: 5px;
        background-color: #1B263B;
        color: #EAEAEA;
        text-align: center;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .file-uploader label:hover {
        background-color: #2E4053;
    }
    .button-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
        gap: 10px;
    }
    .stButton>button {
        width: 150px;
        padding: 10px;
        font-size: 16px;
        color: #EAEAEA;
        background: none;
        border: 2px solid #00A8E8;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00A8E8;
        color: #1B263B;
    }
    .element-container .stButton {
        display: flex;
        justify-content: center;
    }  
    .st-alert {
        border: none;
        padding: 20px;
        font-size: 18px;
        margin-top: 20px;
    }
    .st-success {
        background-color: #1ABC9C;
        color: #ffffff;
    }
    .st-error {
        background-color: #E74C3C;
        color: #ffffff;
    }
    .st-emotion-cache-1jicfl2{
        padding: 0;
    }
    .st-emotion-cache-1jmvea6 {
        color: #00A8E8;
    }
    .st-emotion-cache-1aehpvj{
        color: #00A8E8;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        font-size: 14px;
        color: #EAEAEA;
        padding: 10px 0;
        background-color: #2C3E50;
        animation: fadeIn 2s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideIn {
        from { transform: translateY(-20px); }
        to { transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

# Display logo.
logo_base64 = get_base64_image("logo.png")
logo_html = f"""
    <div class="logo">
        <img src="data:image/png;base64,{logo_base64}" width="110"/>
    </div>
    """
st.markdown(logo_html, unsafe_allow_html=True)

# Title and description
st.markdown('<div class="header">Deepfake Voice Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="info">This website was established by DataPanda AI to address the growing threat of Deepfake technology being used for fraudulent activities.</div>', unsafe_allow_html=True)

# File uploader
st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp_audio_file.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract features from the audio file
    features = extract_features("temp_audio_file.wav")
    
    # Prepare the features for prediction
    features = np.expand_dims(features, axis=0)
    
    # Container for buttons
    st.markdown('<div class="button-container">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        play_audio = st.button('Play Audio')

    with col2:
        start_predict = st.button("Start Predict")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if play_audio:
        st.audio(uploaded_file, format='audio/wav')
    
    if start_predict:
        # Make a prediction using the LSTM model
        prediction = model.predict(features)
        
        # Interpret the prediction result
        if prediction[0][0] > 0.0001: 
            result = 'Real Voice'
            st.markdown(f'<div class="st-alert st-success">The uploaded voice is: {result}</div>', unsafe_allow_html=True)
        else:
            result = 'Ai Voice'
            st.markdown(f'<div class="st-alert st-error">The uploaded voice is: {result}</div>', unsafe_allow_html=True)

        # Output prediction value (optional for debugging)
        # st.write(f"Prediction value: {prediction[0][0]}")

# Footer
st.markdown('<div class="footer">By DataPanda AI 2024</div>', unsafe_allow_html=True)
