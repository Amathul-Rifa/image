import streamlit as st
from PIL import Image
import io
import pandas as pd
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY")

# API URLs
API_URL_GENDER = "https://api-inference.huggingface.co/models/mrm8488/deepface-gender"
API_URL_DETECTOR = "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector"

# Headers
headers = {"Authorization": f"Bearer {api_key}"}

# Query gender model
def query_gender(image):
    image = image.convert('RGB')
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)
    try:
        response = requests.post(API_URL_GENDER, headers=headers, data=image_bytes, timeout=30)
        return response
    except requests.exceptions.Timeout:
        return None

# Query detector model
def query_detector(image_bytes):
    try:
        response = requests.post(API_URL_DETECTOR, headers=headers, data=image_bytes, timeout=30)
        return response.json()
    except requests.exceptions.Timeout:
        return None

# Gender classification page
def gender_classification():
    st.title("🧑 Gender Classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        with st.spinner("Classifying..."):
            response = query_gender(image)

        if response is None:
            st.error("❌ Request timed out. Try again later.")
        elif response.status_code != 200:
            st.error(f"❌ API error. Status code: {response.status_code}")
        else:
            result = response.json()
            st.json(result)

            if isinstance(result, list) and all(isinstance(i, dict) for i in result):
                df = pd.DataFrame(result)
                st.write("API Response Table:")
                st.table(df)
                top_result = df.loc[df['score'].idxmax()]
                label = top_result['label']
                score = top_result['score']
                st.success(f"**Prediction:** {label} with confidence **{score:.2f}**")
            else:
                st.warning("⚠️ Unexpected API response format.")

# AI Image Detector page
def ai_image_detector():
    st.title("🕵️ AI Image Detector")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
        image_bytes = uploaded_file.read()

        with st.spinner("Analyzing..."):
            result = query_detector(image_bytes)

        if result is None:
            st.error("❌ Request timed out. Try again.")
        elif isinstance(result, list) and all(isinstance(i, dict) for i in result):
            st.json(result)
            df = pd.DataFrame(result)
            st.write("API Response Table:")
            st.table(df)

            top_result = df.loc[df['score'].idxmax()]
            label = top_result["label"]
            score = top_result["score"]
            st.success(f"**Prediction:** {label} with confidence **{score:.2f}**")
        else:
            st.warning("⚠️ Unexpected API response format.")

# Is Image Artificial? page
def is_artificial_detector():
    st.title("🤖 Is the Image Artificial?")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
        image_bytes = uploaded_file.read()

        with st.spinner("Analyzing..."):
            result = query_detector(image_bytes)

        if result is None:
            st.error("❌ Request timed out.")
        elif isinstance(result, list):
            is_artificial = any(
                item.get('label', '').lower() == 'artificial' and item.get('score', 0) > 0.20
                for item in result
            )
            st.json(result)
            if is_artificial:
                st.warning("⚠️ The image may be **artificially generated**.")
            else:
                st.success("✅ The image is likely **human-made**.")
        else:
            st.warning("⚠️ Unexpected API response format.")

# Main app
def main():
    st.set_page_config(page_title="AI Image Tools", page_icon="🤖")
    st.sidebar.title("🔍 Navigation")
    choice = st.sidebar.radio("Go to", ["Gender Classification", "AI Image Detector", "Is Image Artificial?"])

    if choice == "Gender Classification":
        gender_classification()
    elif choice == "AI Image Detector":
        ai_image_detector()
    elif choice == "Is Image Artificial?":
        is_artificial_detector()

if __name__ == "__main__":
    main()
