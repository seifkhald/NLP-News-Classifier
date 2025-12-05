"""
News Text Classifier using Natural Language Processing (NLP)
This script loads news data, preprocesses text, and trains a classifier to predict news categories.
"""

import json
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class NewsClassifier:
    """A class to handle news text classification using NLP techniques."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency
            stop_words='english'
        )
        self.label_encoder = LabelEncoder()
        self.model = None
        self.model_name = None
        
    def preprocess_text(self, text):
        """
        Preprocess text by removing special characters, extra spaces, and converting to lowercase.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep spaces and alphanumeric
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_data(self, file_path):
        """
        Load news data from JSON file.
        
        Args:
            file_path (str): Path to the JSON dataset file
            
        Returns:
            pd.DataFrame: DataFrame containing the news data
        """
        print(f"Loading data from {file_path}...")
        data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} news articles.")
        print(f"Categories found: {df['category'].nunique()}")
        print(f"\nCategory distribution:")
        print(df['category'].value_counts().head(10))
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare features by combining headline and description, and preprocessing text.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target labels
        """
        print("\nPreparing features...")
        
        # Combine headline and short_description
        df['text'] = df['headline'].fillna('') + ' ' + df['short_description'].fillna('')
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]
        
        # Prepare features and labels
        X = df['processed_text'].values
        y = df['category'].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Prepared {len(X)} samples with {len(self.label_encoder.classes_)} categories.")
        
        return X, y_encoded
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the classifier on the data.
        
        Args:
            X (array-like): Feature texts
            y (array-like): Target labels
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        print("\nSplitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        print("\nVectorizing text features using TF-IDF...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"Feature matrix shape: {X_train_tfidf.shape}")
        
        # Try multiple models and select the best one
        models = {
            'Multinomial Naive Bayes': MultinomialNB(alpha=0.1),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=-1),
            'Linear SVM': LinearSVC(random_state=random_state, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        }
        
        best_model = None
        best_name = None
        best_score = 0
        
        print("\nTraining and evaluating models...")
        print("=" * 60)
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_tfidf, y_train)
            
            # Evaluate on test set
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{name} - Test Accuracy: {accuracy:.4f}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_name = name
        
        print("=" * 60)
        print(f"\nBest Model: {best_name} with accuracy: {best_score:.4f}")
        
        # Set the best model
        self.model = best_model
        self.model_name = best_name
        
        # Detailed evaluation of best model
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, best_model.predict(X_test_tfidf), 
                                    target_names=self.label_encoder.classes_))
        
        return best_model, best_score
    
    def predict(self, text):
        """
        Predict the category of a news text.
        
        Args:
            text (str): News text to classify
            
        Returns:
            str: Predicted category
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Vectorize
        text_tfidf = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction_encoded = self.model.predict(text_tfidf)[0]
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        return prediction
    
    def predict_proba(self, text, top_n=3):
        """
        Get probability predictions for a news text.
        
        Args:
            text (str): News text to classify
            top_n (int): Number of top predictions to return
            
        Returns:
            list: List of tuples (category, probability) sorted by probability
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Vectorize
        text_tfidf = self.vectorizer.transform([processed_text])
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(text_tfidf)[0]
        else:
            # For SVM, use decision function
            decision_scores = self.model.decision_function(text_tfidf)[0]
            probabilities = np.exp(decision_scores) / np.sum(np.exp(decision_scores))
        
        # Get top N predictions
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        top_predictions = [
            (self.label_encoder.classes_[idx], probabilities[idx])
            for idx in top_indices
        ]
        
        return top_predictions
    
    def save_model(self, filepath='news_classifier_model.pkl'):
        """
        Save the trained model and vectorizer to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'model_name': self.model_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath='news_classifier_model.pkl'):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.model_name = model_data.get('model_name', 'Unknown')
        
        print(f"Model loaded from {filepath}")
        print(f"Model type: {self.model_name}")


def main():
    """Main function to run the news classifier."""
    # Initialize classifier
    classifier = NewsClassifier()
    
    # Load data
    df = classifier.load_data('News_Category_Dataset_v3.json')
    
    # Prepare features
    X, y = classifier.prepare_features(df)
    
    # Train model
    print("\n" + "=" * 60)
    print("TRAINING NEWS CLASSIFIER")
    print("=" * 60)
    classifier.train(X, y)
    
    # Save model
    classifier.save_model('news_classifier_model.pkl')
    
    # Example predictions
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)
    
    example_texts = [
        "Breaking: New COVID-19 vaccine shows promising results in clinical trials",
        "Lakers win championship game against Celtics in overtime",
        "New iPhone release date announced for next month",
        "Climate change summit discusses carbon emission reduction targets",
        "Movie review: Latest action film breaks box office records"
    ]
    
    for text in example_texts:
        prediction = classifier.predict(text)
        top_predictions = classifier.predict_proba(text, top_n=3)
        print(f"\nText: {text}")
        print(f"Predicted Category: {prediction}")
        print("Top 3 Predictions:")
        for cat, prob in top_predictions:
            print(f"  - {cat}: {prob:.4f}")


if __name__ == "__main__":
    main()

