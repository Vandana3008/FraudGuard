import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session management

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize Lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.remove('not')

# Preprocess text function
def preprocess_text(review):
    # Tokenize and convert to lowercase
    tokens = word_tokenize(review.lower())
    # Remove stopwords and non-alphanumeric tokens, and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

# Model paths
MODEL_PATH = 'model/sentiment_model.pkl'
VECTORIZER_PATH = 'model/vectorizer.pkl'

def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        # Load the trained model and vectorizer
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer

# Initialize model and vectorizer
model, vectorizer = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        # Here you would typically validate the credentials against a database
        # For now, we'll just set a session variable
        session['user'] = email
        return redirect(url_for('welcome') + '?login_success=true')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Here you would typically:
        # 1. Validate the form data
        # 2. Check if the email is already registered
        # 3. Hash the password
        # 4. Store the user in the database
        
        # For now, we'll just redirect to login with success message
        return redirect(url_for('login') + '?register_success=true')
    return render_template('register.html')

@app.route('/welcome')
def welcome():
    # Check if user is logged in
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('welcome.html')

@app.route('/fraudguard')
def fraudguard():
    # Check if user is logged in
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('fraudguard.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        app_name = request.form['app_name']
        
        # For now, we'll use a placeholder review based on the app name
        # In a real application, you would fetch reviews for this app from a database or API
        review = f"Reviews for {app_name}"
        
        # Preprocess and predict
        review_processed = preprocess_text(review)
        review_vectorized = vectorizer.transform([review_processed])
        prediction = model.predict(review_vectorized)[0]
        
        # Determine if app is fraud based on sentiment
        is_fraud = prediction == "negative"
        
        result = {
            'app_name': app_name,
            'sentiment': prediction,
            'is_fraud': is_fraud,
            'fraud_percentage': 50  # Placeholder percentage
        }
        
        return render_template('result.html', result=result)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    review = data.get('review', '')
    
    # Preprocess and predict
    review_processed = preprocess_text(review)
    review_vectorized = vectorizer.transform([review_processed])
    prediction = model.predict(review_vectorized)[0]
    
    # Determine if app is fraud based on sentiment
    is_fraud = prediction == "negative"
    
    return jsonify({
        'review': review,
        'sentiment': prediction,
        'is_fraud': is_fraud
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        if 'Review' not in df.columns:
            return jsonify({'error': 'CSV must contain a Review column'}), 400
        
        # Process the first 5 reviews
        reviews = df['Review'].tolist()
        predictions = []
        
        for review in reviews:
            review_processed = preprocess_text(review)
            review_vectorized = vectorizer.transform([review_processed])
            prediction = model.predict(review_vectorized)[0]
            predictions.append(prediction)
        
        # Count positive and negative predictions
        positive_count = predictions.count("positive")
        negative_count = predictions.count("negative")
        
        # Determine if fraud based on the number of positive reviews
        is_fraud = negative_count > positive_count
        fraud_percentage = (negative_count / len(reviews)) * 100  # Calculate fraud percentage
        
        # Get app name from the form
        app_name = request.form.get('app_name')
        if not app_name:
            return jsonify({'error': 'App name is required'}), 400
        
        result = {
            'app_name': app_name,
            'predictions': predictions,
            'is_fraud': is_fraud,
            'fraud_percentage': fraud_percentage
        }
        
        return render_template('result.html', result=result)  # Render result.html with the results
    else:
        return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
