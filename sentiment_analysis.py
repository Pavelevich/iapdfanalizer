import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text):
        sentiment_scores = self.analyzer.polarity_scores(text)

        positive = sentiment_scores['pos']
        neutral = sentiment_scores['neu']
        negative = sentiment_scores['neg']

        total = positive + neutral + negative
        if total > 0:
            positive_percentage = (positive / total) * 100
            neutral_percentage = (neutral / total) * 100
            negative_percentage = (negative / total) * 100
        else:
            positive_percentage = neutral_percentage = negative_percentage = 0


        if positive_percentage > 50:
            sentiment_description = "Predominantly positive"
        elif negative_percentage > 50:
            sentiment_description = "Predominantly negative"
        else:
            sentiment_description = "Neutral"

        return {
            "positive": positive_percentage,
            "neutral": neutral_percentage,
            "negative": negative_percentage,
            "description": sentiment_description,
            "raw_scores": sentiment_scores
        }
