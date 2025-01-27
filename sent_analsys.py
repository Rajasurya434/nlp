from transformers import pipeline


sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result[0]['label'], result[0]['score']


if __name__ == "__main__":
    user_input = input("Enter text for sentiment analysis: ")
    label, score = analyze_sentiment(user_input)
    print(f"Sentiment: {label} with confidence score: {score:.2f}")
