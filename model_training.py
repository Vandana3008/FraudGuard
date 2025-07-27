# model_training.py
import pandas as pd
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.remove('not')  # Ensure 'not' is not removed

# Preprocess text function
def preprocess_text(review):
    # Tokenize and convert to lowercase
    tokens = word_tokenize(str(review).lower())
    # Remove stopwords and non-alphanumeric tokens, and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

# Load and train model function
def load_and_train_model():
    try:
        # Load dataset (make sure the file path is correct)
        df = pd.read_csv('DatasetReviewsAndSentiments.csv')
        print("dataset read")
        
        # Preprocess reviews
        df['Review'] = df['Review'].apply(preprocess_text)
        
        # Vectorize the text data with bi-grams and tri-grams
        vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        X = vectorizer.fit_transform(df['Review'])
        y = df['Label']
        
        # Split data into training and testing sets with stratified sampling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
        
        # Define the model
        model = MultinomialNB()
        
        # Define the parameter grid
        param_grid = {
            'alpha': [0.1, 0.5, 0.7],
            'fit_prior': [True, False]
        }
        
        # Set up GridSearchCV with stratified k-fold cross-validation
        stratified_kfold = StratifiedKFold(n_splits=5)
        grid_search = GridSearchCV(model, param_grid, cv=stratified_kfold, scoring='f1_weighted')
        grid_search.fit(X_train, y_train)
        
        # Make predictions on test set for evaluation
        y_pred = grid_search.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(y_pred, accuracy)
        
        # Save the trained model and vectorizer using pickle
        with open('model/sentiment_model.pkl', 'wb') as model_file:
            pickle.dump(grid_search, model_file)
            print("model saved")
        
        with open('model/vectorizer.pkl', 'wb') as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)
            print("model saved 2")
        
        return grid_search, vectorizer, accuracy
    
    except FileNotFoundError:
        return None, None, None


load_and_train_model()