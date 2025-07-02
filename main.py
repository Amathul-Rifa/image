import streamlit as st
from PIL import Image
import io
import pandas as pd

# API Endpoints
API_URL_GENDER = "https://api-inference.huggingface.co/models/rizandwiki/gender-classification"
API_URL_DETECTOR = "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

import requests
from dotenv import load_dotenv
import os

# Load Hugging Face API Key
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")



# ------------ API Functions ------------

def query_gender(image):
    image = image.convert("RGB")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)
    
    response = requests.post(API_URL_GENDER, headers=headers, data=image_bytes)
    return response


def query_detector(image_bytes):
    response = requests.post(API_URL_DETECTOR, headers=headers, data=image_bytes)
    response.json()


def gender_classification():
    st.header("🚻 Gender Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with st.spinner("Classifying..."):
            response = query_gender(image)

        if isinstance(result, dict) and "error" in result:
            st.error(result["error"])
            if "details" in result:
                st.code(result["details"], language="json")
        elif result and isinstance(result, list):
            df = pd.DataFrame(result)
            st.subheader("Results:")
            st.table(df)

            top_result = df.loc[df['score'].idxmax()]
            st.success(f"Predicted Gender: **{top_result['label']}** (Score: {top_result['score']:.2f})")
        else:
            st.error("Unknown error occurred. Try again with another image.")


def ai_image_detector():
    st.header("🧠 AI Image Detector")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        image_bytes = uploaded_file.read()

        with st.spinner("Detecting..."):
            result = query_detector(image_bytes)

        if result:
            df = pd.DataFrame(result)
            st.subheader("Results:")
            st.table(df)

            if not df.empty:
                top_result = df.loc[df['score'].idxmax()]
                st.success(f"The image is likely **{top_result['label']}** (Score: {top_result['score']:.2f})")
            else:
                st.warning("No meaningful prediction.")
        else:
            st.error("API error. Try again later.")


def is_artificial_detector():
    st.header("🤖 Is Image Artificial?")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        image_bytes = uploaded_file.read()

        with st.spinner("Analyzing..."):
            result = query_detector(image_bytes)

        if result:
            is_artificial = any(
                item['label'] == 'artificial' and item['score'] > 0.20
                for item in result
            )
            if is_artificial:
                st.warning("⚠️ The image may be artificially generated.")
            else:
                st.success("✅ The image is likely human.")
        else:
            st.error("Could not connect to the API.")


# ------------ Main App ------------

def main():
    st.set_page_config(page_title="AI Image Tools", page_icon="🧠")

    st.sidebar.title("🧭 Navigation")
    selection = st.sidebar.radio("Choose a Tool", [
        "Gender Classification",
        "AI Image Detector",
        "Is Image Artificial?"
    ])

    if selection == "Gender Classification":
        gender_classification()
    elif selection == "AI Image Detector":
        ai_image_detector()
    elif selection == "Is Image Artificial?":
        is_artificial_detector()


if __name__ == "__main__":
    main()
