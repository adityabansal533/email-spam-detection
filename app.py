import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page settings
st.set_page_config(page_title="Email Spam Detector", page_icon="📧")

st.title("📧 AI Email Spam Detector")
st.write("Enter an email message below to check whether it is Spam or Not Spam.")

# Input box
email = st.text_area("Email Message")

# Detect button
if st.button("Detect Spam"):
    if email.strip() == "":
        st.warning("Please enter an email message.")
    else:
        data = vectorizer.transform([email])
        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("🚨 Spam Email")
        else:
            st.success("✅ Not a Spam Email")