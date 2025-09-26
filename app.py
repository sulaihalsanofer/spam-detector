pip install joblib
import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit UI
st.title("📩 Spam Message Classifier")
st.write("Enter a message below to check whether it's **Spam** or **Not Spam**")

# Input box
user_input = st.text_area("Your Message", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Transform and predict
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)

        if prediction[0] == 1:
            st.error("This is a SPAM message!")
        else:
            st.success("This is NOT a spam message.")


