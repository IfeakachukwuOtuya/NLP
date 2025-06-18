
import streamlit as st         # For building the web app UI
import nltk                    # Natural Language Toolkit (for NLP tasks)
import re                      # Regular expressions (for pattern matching)
from nltk.tokenize import word_tokenize, sent_tokenize  # Tokenizers
from nltk.corpus import stopwords                        # Common stopwords
from wordcloud import WordCloud                          # WordCloud generator
import matplotlib.pyplot as plt                          # For plotting the word cloud


# Download necessary NLTK data #Download NLTK Resources (Once per run)
nltk.download('punkt')        # Tokenizer models (for sentences and words)
nltk.download('stopwords')    # Common stopwords (like "is", "the", etc.)

# Set of English stopwords
stop_words = set(stopwords.words('english')) # Use a set for faster lookup

# Streamlit App
st.title("NLP Text Analyzer with WordCloud") #page title


# Text input
user_input = st.text_area("Enter your text here:", height=200) # Input box

# Process Text After User Clicks "Analyze"
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.") # Checks if button is clicked and input is not empty.
    else:
        # Tokenization
        words = word_tokenize(user_input)  # Splits into word tokens
        sentences = sent_tokenize(user_input) # Splits into sentence tokens

        # Identify Stopwords
        stopword_tokens = [word for word in words if word.lower() in stop_words] # Filters out words that are in the stopword list.

        # Blank space count (2 or more spaces)
        blank_space_matches = re.findall(r'  +', user_input) 
        blank_spaces_count = len(blank_space_matches) # Finds groups of 2 or more spaces

        # Whitespace character count (spaces, tabs, newlines, etc.)
        whitespace_matches = re.findall(r'\s', user_input)  # All whitespace (spaces, tabs, newlines)
        whitespace_chars_count = len(whitespace_matches) 

        # Display results
        st.subheader("Tokenized Words")
        st.write(words)

        st.subheader("Tokenized Sentences")
        st.write(sentences)

        st.subheader("Stopword Tokens")
        st.write(stopword_tokens)

        st.subheader("Blank Spaces (2 or more spaces together)")
        st.write(f"Found {blank_spaces_count} blank space sequences:")
        st.code(blank_space_matches if blank_space_matches else "None")

        st.subheader("Whitespace Characters (spaces, tabs, newlines)")
        st.write(f"Found {whitespace_chars_count} whitespace characters:")
        st.code(whitespace_matches if whitespace_matches else "None")

        #  Generate and Display Word Cloud 
        st.subheader("Word Cloud")
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              stopwords=stop_words).generate(user_input)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
        # Creates a WordCloud excluding stopwords and displays it using Matplotlib in Streamlit.