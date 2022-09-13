"""
Demo UI page
"""
import streamlit as st
#import tools.ocr_video as ocr
import os
import shutil
import uuid
from model_loader import HFPretrainedModel
from transformers import pipeline
import torch

@st.cache(hash_funcs={"MyUnhashableClass": lambda _: None})
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
    
@st.cache(hash_funcs={"MyUnhashableClass": lambda _: None})
def load_campaign_model():
    return HFPretrainedModel("distilbert-base-uncased", "deano/political-campaign-analysis-110922")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid1()
    
# Temporary folder path
TMP_PATH = "tmp-{"+str(st.session_state["session_id"])+"}/"

st.title("Demo page")
st.markdown("""Upload the US political campaign video to predict its orientation (base/center).""")
video_file = st.file_uploader("Choose the US political campaign video", type=["wmv", "avi", "mov"], disabled=True)
text = st.text_input("Transcript of the video", "")
b = st.button("Predict")
if b:
    st.markdown("""---""")
    status_bar = st.progress(0)
    upload_cap = st.caption("Uploading video...")
    #if os.path.isdir(TMP_PATH):
    #    shutil.rmtree(TMP_PATH)
    #os.mkdir(TMP_PATH)
    #with open(TMP_PATH+"uploaded_video_tmp", "wb") as f:
    #    f.write(video_file.getbuffer())
    status_bar.progress(50)
    #upload_cap.caption("Extracting text from frames... (can take some time)")
    #text_ocr = ocr.get_formated_text(ocr.retrieve_text(TMP_PATH+"uploaded_video_tmp", frames_path = "tmp_frames-{"+str(st.session_state["session_id"])+"}", show_print = False))
    upload_cap.caption("Extracting text sentiment...")
    sentiment_analysis = load_sentiment_model()
    text_sentiment = sentiment_analysis(text)[0]["label"]
    status_bar.progress(80)
    
    #shutil.rmtree(TMP_PATH)
    status_bar.progress(90)
    upload_cap.caption("Prediction...")
    model = load_campaign_model()
    #query_dict = {"text": [text], "text_ocr": [text_ocr]}
    query_dict = {"text": [text], "label_sentiment": [text_sentiment]}
    # Predicted confidence for each label
    conf = model.predict(query_dict)
    col1, col2 = st.columns(2)
    col1.metric("Base", "{:.2f}".format(conf[1].item()*100)+"%", "")
    col2.metric("Center", "{:.2f}".format(conf[0].item()*100)+"%", "")
    
    status_bar.progress(100)
    upload_cap.caption("Done")
