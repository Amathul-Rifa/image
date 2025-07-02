import streamlit as st
from PIL import Image
import io
import pandas as pd
import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("HUGGINGFACE_API_KEY")

# Hugging Face API URLs
API_URL_GENDER = "https://api-inference.huggingface.co/models/rizvandwiki/gender-classification"
API_URL_DETECTOR = "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector"

# Set headers
headers = {"Authorization": f"Bearer {api_key}"}


def query_gender(image):
    image = image.convert('RGB')
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)
    response = requests.post(API_URL_GENDER, headers=headers, data=image_bytes)
    return response

def query_detector(image_bytes):
    response = requests.post(API_URL_DETECTOR, headers=headers, data=image_bytes)
    return response.json()

def gender_classification():
    st.title("Gender Classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        with st.spinner('Classifying...'):
            response = query_gender(image)
        
        st.write("API Response Status Code:", response.status_code)

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                df = pd.DataFrame(result)
                st.write("API Response:")
                st.table(df)

                top_result = df.loc[df['score'].idxmax()]
                label = top_result['label']
                score = top_result['score']
                st.write(f"The person in the image is likely to be **{label}** with a score of **{score:.2f}**.")
            else:
                st.write("Unexpected response format.")
        else:
            st.write("Failed to get a valid response from the API.")

def ai_image_detector():
    st.title("AI Image Detector")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        image_bytes = uploaded_file.read()

        with st.spinner("Analyzing..."):
            result = query_detector(image_bytes)

        if result:
            df = pd.DataFrame(result)
            st.write("API Response:")
            st.table(df)

            if not df.empty:
                top_result = df.loc[df['score'].idxmax()]
                label = top_result["label"]
                score = top_result["score"]
                st.write(f"The image is likely **{label}** with a score of **{score:.2f}**.")
        else:
            st.write("Failed to get a valid response from the API.")

def is_artificial_detector():
    st.title("Is Image Artificial?")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        image_bytes = uploaded_file.read()

        with st.spinner("Analyzing..."):
            result = query_detector(image_bytes)

        if result:
            is_artificial = False
            for item in result:
                if item['label'].lower() == 'artificial' and item['score'] > 0.20:
                    is_artificial = True
                    break
            if is_artificial:
                st.write("🔍 The image may be **artificially generated**.")
            else:
                st.write("✅ The image is likely **human-made**.")
        else:
            st.write("Failed to get a valid response from the API.")

def main():
    st.set_page_config(page_title="AI Image Tools", page_icon=":robot:")
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Gender Classification", "AI Image Detector", "Is Image Artificial?"])

    if selection == "Gender Classification":
        gender_classification()
    elif selection == "AI Image Detector":
        ai_image_detector()
    elif selection == "Is Image Artificial?":
        is_artificial_detector()

if __name__ == "__main__":
    main()
