
import streamlit as st
import pyttsx3
import os

# Streamlit UI

page_bg_img = '''
<style>
.stApp {
  
  background-image: url("https://media.istockphoto.com/id/2204180039/photo/digital-agent-assistant-smart-speaker-visualization-of-ai-as-an-assistant-machine-learning.jpg?s=612x612&w=0&k=20&c=Tiji-8HTGq4i6fDTc5JovNMxOLmcoBOmbtTrgGtK8SE=");
  
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

st.markdown("<h1><span style='font-weight:bold;color:white;'>🎙️ Text-to-Speech App</span></h1>", unsafe_allow_html=True)

st.markdown("### Enter text to convert to speech:")


# User input
text = st.text_area("### Enter text:", height=150)

voice_gender = st.selectbox("Choose voice gender:", ("Male", "Female"))

# Convert to speech when button clicked
if st.button("🔊 Convert & Play"):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.8)

    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id if voice_gender == "Male" else voices[1].id)

    # Save audio
    audio_file = "output.mp3"
    engine.save_to_file(text, audio_file)
    engine.runAndWait()

    # Audio playback
    st.success("✅ Audio generated!")
    audio_bytes = open(audio_file, 'rb').read()
    st.audio(audio_bytes, format='audio/mp3')

    # Clean up after
    os.remove(audio_file)
