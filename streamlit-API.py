import streamlit as st
import requests

st.title("Video Upload to FastAPI")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    st.video(uploaded_file)

    if st.button("Send to API"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://localhost:8000/predict/", files=files)

        if response.status_code == 200:
            st.success(f"Prediction: {response.json()}")
        else:
            st.error("Failed to send video to API")
