import streamlit as st
import joblib
import json
import csv
from datetime import datetime
import os

# Constants for file handling and input validation
HISTORY_FILE = "transaction_history.json"  # File to store transaction history
MAX_HISTORY_ENTRIES = 50  # Maximum number of transactions to keep in history
MIN_INPUT_LENGTH = 10  # Minimum input length for transaction analysis

# Load model and vectorizer
@st.cache_resource  # Caches the model and vectorizer to optimize performance
def load_model():
    try:
        model = joblib.load("fraud_model.pkl")  # Load trained fraud detection model
        vectorizer = joblib.load("vectorizer.pkl")  # Load text vectorizer
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")  # Display error if model loading fails
        return None, None

model, vectorizer = load_model()  # Load the model and vectorizer

# Load transaction history from JSON file
def load_history():
    if os.path.exists(HISTORY_FILE):  # Check if history file exists
        try:
            with open(HISTORY_FILE, "r") as file:
                return json.load(file)  # Load and return history data
        except json.JSONDecodeError:  # Handle JSON decoding errors
            return []
    return []

history = load_history()  # Initialize transaction history

# Save transaction history to JSON file
def save_history(history):
    with open(HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)  # Save history with formatting

# Fraud detection function
def analyze_transaction(text, threshold):
    vectorized_input = vectorizer.transform([text])  # Convert text into numerical format
    prediction = model.predict(vectorized_input)[0]  # Make fraud prediction
    confidence = max(model.predict_proba(vectorized_input)[0])  # Get confidence score
    
    # Store transaction details
    transaction = {
        "text": text,
        "prediction": "FRAUD" if prediction == 1 else "NO_FRAUD",
        "confidence": confidence,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "threshold": threshold
    }
    return transaction

# Streamlit UI setup
st.title("💳 UPI Fraud Shield")  # App title
st.write("Detect potential fraud in UPI transactions using AI.")  # App description

# User input section
text_input = st.text_area("Enter UPI transaction details:")  # Text input for transaction details
threshold = st.slider("Confidence Threshold (%)", 50, 95, 80)  # Confidence threshold slider

# Button to analyze transaction
if st.button("Analyze Transaction"):
    if len(text_input) < MIN_INPUT_LENGTH:
        st.error(f"Please enter at least {MIN_INPUT_LENGTH} characters.")  # Error for short input
    else:
        transaction = analyze_transaction(text_input, threshold / 100)  # Analyze transaction
        
        # Display results
        st.subheader("Results")
        st.write(f"*Prediction:* {transaction['prediction']}")
        st.write(f"*Confidence:* {transaction['confidence']:.2%}")
        
        # Warning if fraud risk is high
        if transaction['prediction'] == "FRAUD" and transaction['confidence'] >= transaction['threshold']:
            st.warning("⚠ High risk detected!")
        
        # Update transaction history
        history.append(transaction)
        if len(history) > MAX_HISTORY_ENTRIES:
            history.pop(0)  # Maintain maximum history size
        save_history(history)  # Save updated history

# Show transaction history
st.subheader("📜 Transaction History")
if history:
    for item in reversed(history):  # Display history in reverse chronological order
        st.text(f"[{item['time']}] {item['prediction']} ({item['confidence']:.2%}) - {item['text']}")
else:
    st.info("No transactions analyzed yet.")  # Message if no history exists

# Export transaction history to CSV
if st.button("Export History as CSV"):
    csv_filename = "transaction_history.csv"
    with open(csv_filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "Prediction", "Confidence", "Threshold", "Text"])  # CSV headers
        for item in history:
            writer.writerow([item['time'], item['prediction'], f"{item['confidence']:.2%}", f"{item['threshold']:.0%}", item['text']])
    st.success(f"Transaction history saved as {csv_filename}.")  # Success message