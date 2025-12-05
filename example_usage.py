"""
Example script demonstrating how to use the trained News Classifier
"""

from news_classifier import NewsClassifier


def example_predictions():
    """Demonstrate how to make predictions with the trained model."""
    
    # Initialize classifier
    classifier = NewsClassifier()
    
    # Load the trained model
    try:
        classifier.load_model('news_classifier_model.pkl')
    except FileNotFoundError:
        print("Model file not found. Please run news_classifier.py first to train the model.")
        return
    
    # Example news texts to classify
    test_texts = [
        {
            "text": "The Los Angeles Lakers defeated the Boston Celtics in a thrilling overtime game last night, winning 112-108.",
            "expected": "SPORTS"
        },
        {
            "text": "President announces new economic policy to address inflation concerns across the nation.",
            "expected": "POLITICS"
        },
        {
            "text": "Apple unveils latest iPhone with advanced AI features and improved battery life.",
            "expected": "TECH"
        },
        {
            "text": "Scientists discover new species in the Amazon rainforest, highlighting biodiversity conservation needs.",
            "expected": "ENVIRONMENT"
        },
        {
            "text": "Award-winning actor stars in new blockbuster movie that breaks box office records.",
            "expected": "ENTERTAINMENT"
        },
        {
            "text": "Hurricane causes widespread damage in coastal regions, emergency services respond.",
            "expected": "U.S. NEWS"
        },
        {
            "text": "International summit addresses climate change and carbon emission reduction targets.",
            "expected": "WORLD NEWS"
        }
    ]
    
    print("=" * 70)
    print("NEWS CLASSIFIER - EXAMPLE PREDICTIONS")
    print("=" * 70)
    
    for i, example in enumerate(test_texts, 1):
        text = example["text"]
        expected = example.get("expected", "Unknown")
        
        # Get prediction
        prediction = classifier.predict(text)
        
        # Get top 3 predictions with probabilities
        top_predictions = classifier.predict_proba(text, top_n=3)
        
        print(f"\nExample {i}:")
        print(f"Text: {text[:80]}...")
        print(f"Predicted Category: {prediction}")
        if expected != "Unknown":
            match = "✓" if prediction == expected else "✗"
            print(f"Expected Category: {expected} {match}")
        print("Top 3 Predictions:")
        for category, probability in top_predictions:
            print(f"  - {category}: {probability:.4f} ({probability*100:.2f}%)")
        print("-" * 70)


def interactive_mode():
    """Interactive mode for classifying custom news text."""
    
    classifier = NewsClassifier()
    
    try:
        classifier.load_model('news_classifier_model.pkl')
    except FileNotFoundError:
        print("Model file not found. Please run news_classifier.py first to train the model.")
        return
    
    print("=" * 70)
    print("NEWS CLASSIFIER - INTERACTIVE MODE")
    print("=" * 70)
    print("Enter news text to classify (or 'quit' to exit):\n")
    
    while True:
        text = input("News Text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not text:
            print("Please enter some text.")
            continue
        
        try:
            prediction = classifier.predict(text)
            top_predictions = classifier.predict_proba(text, top_n=3)
            
            print(f"\nPredicted Category: {prediction}")
            print("Top 3 Predictions:")
            for category, probability in top_predictions:
                print(f"  - {category}: {probability:.4f} ({probability*100:.2f}%)")
            print()
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        example_predictions()
        print("\n" + "=" * 70)
        print("To try interactive mode, run: python example_usage.py --interactive")
        print("=" * 70)

