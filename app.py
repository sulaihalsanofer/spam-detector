import streamlit as st
import pickle

# Load the saved model and vectorizer
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

st.title("Spam Message Classifier")

user_input = st.text_area("Enter your message:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a message!")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)
        if prediction[0] == 1:
            st.error("This is a SPAM message!")
        else:
            st.success("This is NOT a spam message.")






