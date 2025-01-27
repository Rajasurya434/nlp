from transformers import pipeline


sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", return_all_scores=True)

def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result[0]  


if __name__ == "__main__":
    user_input = input("Enter text for sentiment analysis: ")
    sentiment = analyze_sentiment(user_input)
    for label in sentiment:
        print(f"Sentiment: {label['label']} with confidence score: {label['score']:.2f}")
