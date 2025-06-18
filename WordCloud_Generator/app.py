

import streamlit as st     # For building the web app UI
from wordcloud import WordCloud  # WorldCloud generator
import matplotlib.pyplot as plt  # For plotting the world cloud
import nltk     # Natural Language Toolkit (for NLP tasks)
from nltk.stem import WordNetLemmatizer, PorterStemmer
import spacy

# Download necessary NLTK resources
nltk.download('punkt')   # Tokenizer models (for sentences and words)
nltk.download('wordnet')

# Ensure spaCy model is downloaded
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def process_text(text, lemmatize=False, stem=False):
    words = nltk.word_tokenize(text)
    if stem:
        words = [stemmer.stem(word) for word in words]
    elif lemmatize:
        words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def named_entity_recognition(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

page_bg_img = '''
<style>
.stApp {
  
  background-image: url("https://images.unsplash.com/photo-1739659153856-ae22a5d2390a?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxmZWF0dXJlZC1waG90b3MtZmVlZHw1MDR8fHxlbnwwfHx8fHw%3D");
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  background-attachment: fixed;
}
h1 {
  font-weight: bold !important;
  color: orange !important;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("<h1><span style='font-weight:bold;color:orange;'>Word cloud</span> Generator  Analyzer</h1>", unsafe_allow_html=True)

st.markdown("### Enter text below to generate a word cloud and perform NLP tasks.")

input_text = st.text_area("Enter your text for word cloud and NER:", height=150)

lemmatize_option = st.checkbox("Lemmatization for Word Cloud")
stem_option = st.checkbox("Stemming for Word Cloud")

if lemmatize_option and stem_option:
    st.warning("Please select only one of 'Lemmatization' or 'Stemming' for word cloud.")

if st.button("Generate Word Cloud"):
    if input_text.strip():
        processed_text = process_text(input_text, lemmatize=lemmatize_option and not stem_option, stem=stem_option and not lemmatize_option)

        st.subheader("Processed Text for Word Cloud:")
        st.write(processed_text)

        wordcloud = WordCloud(width=480, height=480, margin=0, background_color='white').generate(processed_text)

        plt.figure(figsize=(6, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.margins(x=0, y=0)
        st.pyplot(plt)

        entities = named_entity_recognition(input_text)
        if entities:
            st.subheader("Named Entities:")
            for ent_text, ent_label in entities:
                st.write(f"{ent_text}: {ent_label}")
        else:
            st.write("No named entities found. Please try with different text.")

st.markdown("---")
st.markdown("### Enter text to convert:")

convert_text = st.text_area("Enter text to convert:", height=120)

convert_option = st.radio("Convert using:", ("Lemmatizer", "Stemmer"))

if st.button("Convert Text"):
    if convert_text.strip():
        if convert_option == "Lemmatizer":
            converted = process_text(convert_text, lemmatize=True)
        else:
            converted = process_text(convert_text, stem=True)
        st.subheader("Converted Text:")
        st.write(converted)

# Sample text button
if st.button("Use Sample Text for Word Cloud"):
    sample_text = "Hi, I am Ifeakachukwu please  Add ur text"
    st.text_area("Enter your text for word cloud and NER:", value=sample_text, height=150)

st.markdown("Made by IFEAKACHUKWU OTUYA")