# streamlit_app.py

import streamlit as st
import xml.etree.ElementTree as ET
import re
from bs4 import BeautifulSoup

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
    .title {{
        color: #FF8C00;
        font-size: 3em;
        text-align: center;
    }}
    .neon-text {{
        color: #39ff14;
        font-size: 1.5em;
        text-align: center;
        text-shadow: 0 0 5px #39ff14, 0 0 10px #39ff14, 0 0 20px #39ff14, 0 0 40px #39ff14;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


st.title("📄 XML/HTML Text Extractor")

st.write("""
Upload an XML or HTML file to parse its text, clean it, and display the results.
""")

uploaded_file = st.file_uploader("Choose an XML or HTML file", type=['xml', 'html'])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    # Detect file type from extension or content
    file_name = uploaded_file.name.lower()

    if file_name.endswith('.xml'):
        try:
            tree = ET.ElementTree(ET.fromstring(file_bytes.decode('utf-8')))
            root = tree.getroot()
            raw_text = ET.tostring(root, encoding='utf8').decode('utf8')
            cleaned_text = denoise_text(raw_text)
            st.subheader("✅ Extracted & Cleaned Text")
            st.write(cleaned_text)
        except Exception as e:
            st.error(f"Error parsing XML: {e}")

    elif file_name.endswith('.html'):
        try:
            html_text = file_bytes.decode('utf-8')
            cleaned_text = denoise_text(html_text)
            st.subheader("✅ Extracted & Cleaned Text")
            st.write(cleaned_text)
        except Exception as e:
            st.error(f"Error parsing HTML: {e}")

    else:
        st.warning("Unsupported file type. Please upload an XML or HTML file.")


# Signature
st.markdown('<h5 class="neon-text">Made by Ifeakachukwu Otuya</h5>', unsafe_allow_html=True)