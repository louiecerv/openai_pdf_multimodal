import os
import base64
import io
from io import BytesIO
import tempfile
import shutil
import streamlit as st
from PIL import Image
import fitz  # PyMuPDF
from openai import OpenAI

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def extract_text_and_images(file_path):
    text_content = ""
    image_urls = []

    try:
        extension = os.path.splitext(file_path)[1].lower()

        if extension == ".pdf":
            doc = fitz.open(file_path)
            for page_index in range(len(doc)):
                page = doc.load_page(page_index)
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(BytesIO(image_bytes))
                    image.thumbnail((512, 512))

                    buffered = io.BytesIO()
                    image.save(buffered, format="jpeg") # Force JPEG for PDF images
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    data_url = f"data:image/jpeg;base64,{img_str}"
                    image_urls.append(data_url)

                text_content += page.get_text("text") or ""

        elif extension in (".jpg", ".jpeg", ".png"):
            image = Image.open(file_path)
            image.thumbnail((512, 512))

            buffered = io.BytesIO()
            image_format = "jpeg" if extension in (".jpg", ".jpeg") else "png"
            image.save(buffered, format=image_format)

            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_urls.append(f"data:image/{image_format};base64,{img_str}")

        else:
            st.error(f"Unsupported file type: {extension}")

    except Exception as e:
        st.error(f"An error occurred during file processing: {e}")

    return text_content, image_urls

def generate_ai_response(text_content, image_urls, text_prompt):
    try:
        if image_urls:
           messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        *[{"type": "image_url", "image_url": {"url": url}} for url in image_urls]
                    ]
                }
            ]
        
        else:
            messages = [{"role": "user", "content": f"{text_prompt} Analyze the text: {text_content}"}]

        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, max_tokens=2048, stream=True
        )
        return response

    except Exception as e:
        st.error(f"An error occurred during AI response generation: {e}")
        return None

def main():
    text_content = ""
    image_urls = []

    st.title("Multimodal File Processing using GPT-4 Turbo Model")

    uploaded_file = st.file_uploader("Upload a File (PDF, JPG, PNG, JPEG)", type=None)
    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        text_content, image_urls = extract_text_and_images(file_path)

        if text_content:
            st.subheader("Extracted Text")
            st.text(text_content)

        if image_urls:
            st.subheader("Extracted Images")
            for img_url in image_urls:
                st.image(img_url, caption="Extracted Image", use_container_width=True)

        shutil.rmtree(temp_dir)

    text_prompt = st.text_area("Enter a text prompt for the AI model:", "")

    if st.button("Generate Response"):
        if not text_prompt:
            st.warning("Please enter a text prompt.")
            return

        response_placeholder = st.empty()
        response_text = ""

        with st.spinner("Processing..."):
            response = generate_ai_response(text_content, image_urls, text_prompt)

            if response is None:
                st.error("There was an issue contacting the OpenAI API. Please check your API key and try again.")
                return

            for chunk in response:
                if chunk.choices[0].delta.content:
                    delta_content = chunk.choices[0].delta.content
                    response_text += delta_content
                    response_placeholder.write(response_text)

        st.success("Response generated successfully!")

if __name__ == "__main__":
    main()