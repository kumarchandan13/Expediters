import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Sample dataset (replace with actual UPI transaction data)
data = {
    "text": [
        "Paid ₹500 to XYZ Retail via UPI",
        "Received ₹2000 from ABC Pvt Ltd",
        "Suspicious transaction detected",
        "Payment failed due to security reasons",
        "Fraudulent transaction alert",
        "Scam detected: Fake payment request",
        "Verified transaction: ₹1000 sent",
        "Unauthorized payment attempt blocked"
    ],
    "label": [0, 0, 1, 1, 1, 1, 0, 1]  # 0 = No Fraud, 1 = Fraud
}

df = pd.DataFrame(data)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = np.array(df["label"])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save Model & Vectorizer
joblib.dump(model, "fraud_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Model and vectorizer saved!")