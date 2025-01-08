import os
import base64
import io
from io import BytesIO

import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from openai import OpenAI

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def extract_text_and_images_from_pdf(pdf_file):
    try:
        text_content = ""
        image_urls = []

        pdf_stream = BytesIO(pdf_file.read())

        # Extract text using PdfReader
        pdf_reader = PdfReader(pdf_stream)
        for page in pdf_reader.pages:
            text_content += page.extract_text()

        # Extract images using PyMuPDF
        doc = fitz.open(stream=pdf_stream)
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(BytesIO(image_bytes))
                # Resize image (optional)
                image.thumbnail((512, 512))  # Adjust size as needed

                # Encode the image as base64 and create a data URL
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                data_url = f"data:image/jpeg;base64,{img_str}"
                image_urls.append(data_url)

        return text_content, image_urls
    except Exception as e:
        st.error(f"An error occurred during PDF processing: {e}")
        return "", []

def generate_ai_response(text_content, image_urls, text_prompt):
    try:
        # Construct the messages list with the prompt and base64-encoded image URLs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    *[{"type": "image_url", "image_url": {"url": url}} for url in image_urls]
                ]
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=2048,
        )

        content_string = response.choices[0].message.content
        return content_string
    except Exception as e:
        st.error(f"An error occurred during AI response generation: {e}")
        return ""

def main():
    st.title("Multimodal PDF Processing using GPT-4 Turbo Model")

    text = """Prof. Louie F. Cervantes, M. Eng. (Information Engineering)
    CCS 229 - Intelligent Systems
    Department of Computer Science
    College of Information and Communications Technology
    West Visayas State University
    """
    with st.expander("About"):
        st.text(text)

    st.write("Upload a PDF file for analysis.")

    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_pdf is not None:
        text_content, image_urls = extract_text_and_images_from_pdf(uploaded_pdf)

        st.subheader("Extracted Text")
        st.text(text_content)

        text_prompt = st.text_area("Enter a text prompt for the AI model:", "")

        if image_urls:
            st.subheader("Extracted Images")
            for img_url in image_urls:
                st.image(img_url, caption="Extracted Image", use_container_width=True)

        if st.button("Generate Response"):
            with st.spinner("Processing..."):
                ai_response = generate_ai_response(text_content, image_urls, text_prompt)
                st.success("Response generated!")
                st.markdown(f"AI Response: {ai_response}")

if __name__ == "__main__":
    main()