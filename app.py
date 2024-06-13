from transformers import pipeline
import requests
import os
import streamlit as st
import tempfile
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import langchain_community  # Ensure this line is included if langchain_community modules are used

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

os.environ['OPENAI_API_KEY'] = 'sk-proj-a8YGaUbymvMc81FPWvSAT3BlbkFJgBeDWVKPdfbe5vP7X4R0'


# Image to Text Generation
def img2text(url):
    try:
        image_to_text = pipeline('image-to-text', model="Salesforce/blip-image-captioning-base", max_new_tokens=100)
        text = image_to_text(url)
        return text[0]["generated_text"]
    except Exception as e:
        st.error(f"Error in image to text generation: {e}")
        return None


# Text to Story Generation
def generate_story(scenario):
    try:
        template = """
        You are a story teller
        You can generate a short story based on a simple narrative, the story should be no more than 100 words:

        CONTEXT: {scenario}
        STORY:
        """
        prompt = PromptTemplate(input_variables=["scenario"], template=template)
        chain = LLMChain(llm=OpenAI(temperature=1), prompt=prompt)
        story = chain.run(scenario)
        return story
    except Exception as e:
        st.error(f"Error in story generation: {e}")
        return None


# Story to Speech Generation
def text2speech(message):
    try:
        API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
        headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
        payloads = {"inputs": message}

        response = requests.post(API_URL, headers=headers, json=payloads)
        with open('audio.mp3', 'wb') as file:
            file.write(response.content)
    except requests.exceptions.RequestException as e:
        st.error(f"Error in text to speech generation: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")


# Integration with Streamlit
def main():
    st.header("Turn _Images_ into Audio :red[Stories]")

    uploaded_file = st.file_uploader("Choose an image..", type='jpg')

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with tempfile.NamedTemporaryFile(delete=False) as file:
            file.write(bytes_data)
            file_path = file.name

        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        scenario = img2text(file_path)
        if scenario:
            story = generate_story(scenario)
            if story:
                text2speech(story)

                with st.expander("Scenario"):
                    st.write(scenario)
                with st.expander("Story"):
                    st.write(story)

                st.audio("audio.mp3")


if __name__ == "__main__":
    main()
