# streamlit_app.py

import streamlit as st
import xml.etree.ElementTree as ET
import re
from bs4 import BeautifulSoup
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import textwrap

# ---------------------------
# Utility functions
# ---------------------------

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub(r'\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = re.sub(r'  ', '', text)
    return text

def create_pdf(text):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Split text into lines and wrap long lines
    lines = text.split('\n')
    y = height - 50  # Initial Y position

    for line in lines:
        wrapped_lines = textwrap.wrap(line, width=100)  # Wrap at ~100 chars
        for wrapped_line in wrapped_lines:
            c.drawString(50, y, wrapped_line)
            y -= 15
            if y < 50:
                c.showPage()
                y = height - 50

    c.save()
    buffer.seek(0)
    return buffer

# ---------------------------
# Streamlit App
# ---------------------------

background_image_url = "https://media.istockphoto.com/id/1058936698/photo/xml-white-3d-write-at-red-wall-3d-rendering.jpg?s=612x612&w=0&k=20&c=90OfYMAny42c3SyJRYBxFLxZFH0nCEPKr5WvYVbi7l8="
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('{background_image_url}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📄 XML/HTML Text Extractor & PDF Exporter")
st.write("""
Upload an XML or HTML file to parse its text, clean it, view the results, and download it as a beautiful PDF.
""")

uploaded_file = st.file_uploader("Choose an XML or HTML file", type=['xml', 'html'])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name.lower()

    if file_name.endswith('.xml'):
        try:
            tree = ET.ElementTree(ET.fromstring(file_bytes.decode('utf-8')))
            root = tree.getroot()
            raw_text = ET.tostring(root, encoding='utf8').decode('utf8')
            cleaned_text = denoise_text(raw_text)

            st.subheader("✅ Extracted & Cleaned Text")
            st.write(cleaned_text)

            pdf_buffer = create_pdf(cleaned_text)
            st.download_button(
                label="📥 Download Clean Text as PDF",
                data=pdf_buffer,
                file_name="cleaned_text.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Error parsing XML: {e}")

    elif file_name.endswith('.html'):
        try:
            html_text = file_bytes.decode('utf-8')
            cleaned_text = denoise_text(html_text)

            st.subheader("✅ Extracted & Cleaned Text")
            st.write(cleaned_text)

            pdf_buffer = create_pdf(cleaned_text)
            st.download_button(
                label="📥 Download Clean Text as PDF",
                data=pdf_buffer,
                file_name="cleaned_text.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Error parsing HTML: {e}")

    else:
        st.warning("Unsupported file type. Please upload an XML or HTML file.")
