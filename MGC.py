#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import os
from keras.layers import Bidirectional, LSTM
st.set_page_config(page_title="Music Genre Prediction", layout="centered")


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('MusicGenre_Model.h5')

    return model

model = load_model()

def preprocess_audio(file):
    signal, sample_rate = librosa.load(file, sr=22050)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T
    mfcc = np.expand_dims(mfcc, axis=-1)
    expected_shape = (130, 13, 1)
    if mfcc.shape[0] < expected_shape[0]:
        mfcc = np.pad(mfcc, ((0, expected_shape[0] - mfcc.shape[0]), (0, 0), (0, 0)), mode='constant')
    mfcc = np.resize(mfcc, expected_shape)
    mfcc = np.expand_dims(mfcc, axis=0)

    return mfcc

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f8ff;
        color: #333;
    }
    .title {
        text-align: center;
        font-size: 2em;
        color: #ff6347;
    }
    .genre {
        font-size: 1.5em;
        font-weight: bold;
        color: #4caf50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🎶 Music Genre Prediction 🎶</div>', unsafe_allow_html=True)
def extract_genre_from_filename(filename):

    genre = os.path.basename(filename).split('.')[0]
    return genre
uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"], label_visibility="collapsed")

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')


    mfcc_preprocessed = preprocess_audio(uploaded_file)
    prediction = model.predict(mfcc_preprocessed)
    predicted_class = np.argmax(prediction, axis=1)
    actual_genre = extract_genre_from_filename(uploaded_file.name)
    genre_labels = ["classical", "jazz", "country", "pop", "rock", "metal", "disco", "hiphop", "reggae", "blues"]
    predicted_genre = genre_labels[predicted_class[0]]

    st.markdown(f'<p class="genre">Predicted Genre: {predicted_genre}</p>', unsafe_allow_html=True)


















