import os
import base64
import requests
import streamlit as st
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
from PyPDF2 import PdfReader
from PIL import Image
import fitz  # PyMuPDF

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")  # Ensure this environment variable is set

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

def extract_text_and_images_from_pdf(pdf_file):
    """
    Extracts text and images from a PDF file.

    Args:
        pdf_file (UploadedFile): The uploaded PDF file.

    Returns:
        tuple: A tuple containing the extracted text and images.
    """
    text_content = ""
    images = []

    # Convert UploadedFile to BytesIO for compatibility
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
            # Convert image bytes to a PIL Image object
            image = Image.open(BytesIO(image_bytes))
            images.append(image)

    return text_content, images

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

    # File upload for PDF
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_pdf is not None:
        text_content, images = extract_text_and_images_from_pdf(uploaded_pdf)

        # Display extracted text
        st.subheader("Extracted Text")
        st.text(text_content)

        # Display extracted images
        if images:
            st.subheader("Extracted Images")
            for img in images:
                st.image(img, caption="Extracted Image", use_container_width=True)

        # Prepare the multimodal payload
        payload = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_content}
                        # Images can be added here if extracted
                    ]
                }
            ],
            "max_tokens": 2048,
        }

        if st.button("Generate Response"):
            with st.spinner("Processing..."):
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

                if response.status_code != 200:
                    st.error(f"Error: {response.status_code} - {response.text}")
                else:
                    content = response.json()
                    content_string = content['choices'][0]['message']['content']
                    st.success("Response generated!")
                    st.markdown(f"AI Response: {content_string}")

if __name__ == "__main__":
    main()
