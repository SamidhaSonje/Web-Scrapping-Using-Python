import pandas as pd
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure required resources are downloaded
nltk.download('twitter_samples', quiet=True)
nltk.download('stopwords', quiet=True)

def load_training_data():
    """
    Load and preprocess training data for sentiment analysis.
    You can replace this with your actual dataset path.
    """
    data_path = os.path.join(os.path.dirname(__file__), "data", "twitter_training_data.csv")
    
    if not os.path.exists(data_path):
        print("⚠️ Training data not found. Creating sample dataset...")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        sample_data = {
            "text": [
                "I love this movie!", "This is terrible...", "Absolutely fantastic experience",
                "Not my taste", "Pretty good overall", "Awful and disappointing",
                "Superb!", "Boring and too long"
            ],
            "label": [
                "positive", "negative", "positive", "negative",
                "positive", "negative", "positive", "negative"
            ]
        }
        df = pd.read_csv(data_path)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
    return df

# --- Sentiment Analysis with VADER ---
def analyze_sentiment_vader(texts):
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for text in texts:
        score = analyzer.polarity_scores(str(text))['compound']
        if score >= 0.05:
            sentiment = 'Positive'
        elif score <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        results.append((text, score, sentiment))
    return pd.DataFrame(results, columns=["Text", "Compound", "Sentiment"])


# --- Train Logistic Regression Model ---
def train_sentiment_model():
    from nltk.corpus import twitter_samples
    pos_tweets = twitter_samples.strings('positive_tweets.json')
    neg_tweets = twitter_samples.strings('negative_tweets.json')

    df = pd.DataFrame({
        'text': pos_tweets + neg_tweets,
        'sentiment': [1]*len(pos_tweets) + [0]*len(neg_tweets)
    })
    return df

# --- Train a Model Dynamically ---
def train_sentiment_model(model_type="Logistic Regression"):
    df = load_training_data()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select model type
    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "Naive Bayes":
        model = MultinomialNB()
    elif model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "SVM":
        model = LinearSVC()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'{model_type} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"outputs/confusion_matrix_{model_type.replace(' ', '_')}.png")
    plt.close()

    return model, vectorizer, accuracy, report, f"outputs/confusion_matrix_{model_type.replace(' ', '_')}.png"


# --- Predict New Tweets ---
def predict_sentiment(model, vectorizer, texts):
    X_new = vectorizer.transform(texts)
    preds = model.predict(X_new)
    return ['Positive' if p == 1 else 'Negative' for p in preds]
