import streamlit as st
from transformers import pipeline

# Load the pre-trained sentiment-analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result[0]['label'], result[0]['score']

st.title("Sentiment Analysis Web App")

user_input = st.text_area("Enter text for sentiment analysis:")

if user_input:
    label, score = analyze_sentiment(user_input)
    st.write(f"Sentiment: {label} with confidence score: {score:.2f}")
