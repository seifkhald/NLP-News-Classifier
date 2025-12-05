# News Text Classifier using NLP

A Natural Language Processing (NLP) based news text classifier that automatically categorizes news articles into different categories using machine learning techniques.

## Features

- **Text Preprocessing**: Cleans and normalizes news text by removing URLs, special characters, and extra whitespace
- **TF-IDF Vectorization**: Converts text into numerical features using Term Frequency-Inverse Document Frequency
- **Multiple ML Models**: Tests and selects the best model from:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Linear Support Vector Machine (SVM)
  - Random Forest Classifier
- **Model Evaluation**: Provides detailed classification reports and accuracy metrics
- **Prediction Interface**: Easy-to-use functions for predicting news categories

## Dataset

The classifier uses the `News_Category_Dataset_v3.json` file which contains over 200,000 news articles with the following fields:
- `headline`: News headline
- `short_description`: Brief description of the article
- `category`: News category (target variable)
- `authors`: Article authors
- `date`: Publication date
- `link`: Article URL

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the main script to train the classifier:
```bash
python news_classifier.py
```

This will:
1. Load the news dataset
2. Preprocess the text data
3. Train multiple models and select the best one
4. Evaluate the model performance
5. Save the trained model to `news_classifier_model.pkl`
6. Show example predictions

### Using the Trained Model

```python
from news_classifier import NewsClassifier

# Initialize classifier
classifier = NewsClassifier()

# Load trained model
classifier.load_model('news_classifier_model.pkl')

# Predict category for a news text
text = "Breaking: New technology breakthrough in artificial intelligence"
prediction = classifier.predict(text)
print(f"Predicted Category: {prediction}")

# Get top 3 predictions with probabilities
top_predictions = classifier.predict_proba(text, top_n=3)
for category, probability in top_predictions:
    print(f"{category}: {probability:.4f}")
```

### Training Your Own Model

```python
from news_classifier import NewsClassifier

# Initialize classifier
classifier = NewsClassifier()

# Load your data
df = classifier.load_data('News_Category_Dataset_v3.json')

# Prepare features
X, y = classifier.prepare_features(df)

# Train the model
classifier.train(X, y)

# Save the model
classifier.save_model('my_model.pkl')
```

## Model Performance

The script automatically tests multiple classification algorithms and selects the best performing one based on accuracy. The evaluation includes:
- Overall accuracy score
- Per-category precision, recall, and F1-score
- Confusion matrix

## Text Preprocessing

The classifier performs the following preprocessing steps:
1. Converts text to lowercase
2. Removes URLs and email addresses
3. Removes special characters (keeps alphanumeric and spaces)
4. Removes extra whitespace
5. Combines headline and short description for richer features

## Features

- **TF-IDF Vectorization**: Uses unigrams and bigrams with:
  - Maximum 10,000 features
  - Minimum document frequency of 2
  - Maximum document frequency of 95%
  - English stop words removal

## Output

The trained model is saved as `news_classifier_model.pkl` which includes:
- The trained classifier model
- TF-IDF vectorizer
- Label encoder for categories

## Example Categories

The classifier can predict various news categories such as:
- U.S. NEWS
- WORLD NEWS
- POLITICS
- SPORTS
- ENTERTAINMENT
- TECH
- CULTURE & ARTS
- ENVIRONMENT
- COMEDY
- PARENTING
- And more...

## Requirements

- Python 3.7+
- numpy
- pandas
- scikit-learn

## License

This project is for educational purposes.

