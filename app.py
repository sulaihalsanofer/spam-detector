import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

@st.cache_data(show_spinner=False)
def load_and_train_model():
    # Load data
    url = "https://raw.githubusercontent.com/codebasics/py/master/ML/14_naive_bayes/spam.csv"
    df = pd.read_csv(url, encoding='latin-1')
    df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, random_state=42)

    # Vectorize and train
    vectorizer = CountVectorizer()
    X_train_count = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_count, y_train)

    return model, vectorizer

st.title("Simple Spam Message Classifier")

model, vectorizer = load_and_train_model()

user_input = st.text_area("Enter your message here:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a message!")
    else:
        vect_input = vectorizer.transform([user_input])
        prediction = model.predict(vect_input)

        if prediction[0] == 1:
            st.error("This is a SPAM message!")
        else:
            st.success("This is NOT a spam message.")








