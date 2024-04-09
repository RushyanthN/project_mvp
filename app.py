import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import re
import os
import io
import altair as alt
import streamlit as st
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables
load_dotenv()

# Import custom modules
from utils.b2 import B2

# App constants
REMOTE_DATA = 'Apple-Twitter-Sentiment-DFE_encoded11.csv'

# Load Backblaze connection
b2 = B2(endpoint=os.environ['B2_ENDPOINT'],
        key_id=os.environ['B2_KEYID'],
        secret_key=os.environ['B2_APPKEY'])

# Caching functions
@st.cache_data
def get_data():
    b2.set_bucket(os.environ['B2_BUCKETNAME'])
    df_apple = b2.get_df(REMOTE_DATA)
    df_apple['date'] = pd.to_datetime(df_apple['date'], format='%a %b %d %H:%M:%S %z %Y')
    df_apple['day_month_year'] = df_apple['date'].dt.strftime('%d/%m/%Y')
    df_apple['cleaned_text'] = df_apple['text'].apply(clean_text)
    return df_apple[['cleaned_text', 'sentiment']]

# Clean text function
def clean_text(text):
    text = ' '.join(word for word in text.split() if not word.startswith('@'))
    text = re.sub(r'@\[A-Za-z0-9\]+', ' ', text)
    text = re.sub(r'http\\S+', ' ', text)
    text = re.sub(r'\[^A-Za-z0-9\\s\]+', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    text = text.lower()
    return text

# Train the model
@st.cache_resource
def train_model():
    df_apple = get_data()
    X_train, X_test, y_train, y_test = train_test_split(df_apple['cleaned_text'], df_apple['sentiment'], test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    model = SVC(kernel='linear')
    model.fit(X_train_vectorized, y_train)
    
    return model, vectorizer

# Streamlit app
st.title("Apple Product Sentiment Analysis")

# Get data and train model
df_apple = get_data()
model, vectorizer = train_model()

# User input
user_input = st.text_area("Enter text related to Apple products:")

# Sentiment analysis
if st.button("Analyze Sentiment"):
    if user_input:
        cleaned_text = clean_text(user_input)
        X_new = vectorizer.transform([cleaned_text])
        predicted_sentiment = model.predict(X_new)[0]
        if predicted_sentiment == 'positive':
            st.write("The sentiment is **Positive**.")
        elif predicted_sentiment == 'negative':
            st.write("The sentiment is **Negative**.")
        else:
            st.write("The sentiment is **Neutral**.")
    else:
        st.warning("Please enter some text to analyze.")


issues_text = """
Issues:

The dataset seems to be biased around Neutral sentiment, So there is some trouble around the output.
"""

next_steps_text = """
Next steps:

Edit the code with suitable comments and implementing the code in Python Class and separating the training of the model into separate .py file
"""

st.subheader("Issues")
st.write(issues_text)

st.subheader("Next Steps")
st.write(next_steps_text)