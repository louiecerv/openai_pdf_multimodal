import os
import base64
import io
from io import BytesIO
import tempfile
import shutil
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from openai import OpenAI

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def extract_text_and_images_from_pdf(pdf_file_path):
    try:
        text_content = ""
        image_urls = []

        # Extract text using PdfReader
        pdf_reader = PdfReader(pdf_file_path)
        for page in pdf_reader.pages:
            text_content += page.extract_text() or ""

        # Extract images using PyMuPDF
        doc = fitz.open(pdf_file_path)
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
        # Construct the messages list
        if image_urls:
            messages = [
                {"role": "user", "content": f"{text_prompt} (Analyze the following text and images)"}
            ]
        else:
            messages = [
                {"role": "user", "content": f"{text_prompt} Analyze the text: {text_content}"}
            ]

        # Create a streaming response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=2048,
            stream=True,
        )
        return response

    except Exception as e:
        st.error(f"An error occurred during AI response generation: {e}")

def main():
    text_content = ""
    image_urls = []

    st.title("Multimodal PDF Processing using GPT-4 Turbo Model")

    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_pdf is not None:
        # Save the uploaded PDF to a temporary directory
        temp_dir = tempfile.mkdtemp()
        pdf_file_path = os.path.join(temp_dir, uploaded_pdf.name)
        with open(pdf_file_path, "wb") as f:
            f.write(uploaded_pdf.getvalue())

        text_content, image_urls = extract_text_and_images_from_pdf(pdf_file_path)

        st.subheader("Extracted Text")
        st.text(text_content)

        if image_urls:
            st.subheader("Extracted Images")
            for img_url in image_urls:
                st.image(img_url, caption="Extracted Image", use_container_width=True)

        # Clean up the temporary directory
        shutil.rmtree(temp_dir)

    text_prompt = st.text_area("Enter a text prompt for the AI model:", "")

    if st.button("Generate Response"):

        response_placeholder = st.empty()
        response_text = ""

        with st.spinner("Processing..."):
            response = generate_ai_response(text_content, image_urls, text_prompt)
            print(response)
            
            # Process and stream the response chunks as they arrive
            for chunk in response:
                if chunk.choices[0].delta.content:
                    delta_content = chunk.choices[0].delta.content
                    response_text += delta_content
                    response_placeholder.write(response_text)

            st.success("Response generated successfully!")


if __name__ == "__main__":
    main()
