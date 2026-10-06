import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("AI Document Risk Classifier")

text = st.text_area("Enter Document Text")

if st.button("Check Risk"):

    data = vectorizer.transform([text])

    prediction = model.predict(data)
    probability = model.predict_proba(data)

    confidence = max(probability[0]) * 100

    st.success(f"Risk Category: {prediction[0]}")
    st.write(f"Confidence: {confidence:.2f}%")

    if confidence < 70:
        st.warning("⚠ Manual Review Required")
    else:
        st.success("✅ Classification Accepted")