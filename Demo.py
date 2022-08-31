import streamlit as st
import tools.ocr_video as ocr
import os
import shutil
# Temporary folder path
TMP_PATH = "tmp/"


st.title("Demo page")
st.markdown("""Upload the US political campaign video to predict its orientation (base/center).""")
video_file = st.file_uploader("Choose the US political campaign video", type=["wmv", "avi", "mov"])
text = st.text_input("Speech-to-text", "")
if video_file is not None:
    status_bar = st.progress(0)
    upload_cap = st.caption("Uploading video...")
    if os.path.isdir(TMP_PATH):
        shutil.rmtree(TMP_PATH)
    os.mkdir(TMP_PATH)
    with open(TMP_PATH+"uploaded_video_tmp", "wb") as f:
        f.write(video_file.getbuffer())
    status_bar.progress(50)
    upload_cap.caption("Extracting text from frames...")
    text_ocr = ocr.retrieve_text(TMP_PATH+"uploaded_video_tmp", show_print = False)
    text_feat = " ".join(text_ocr)
    status_bar.progress(80)
    
    shutil.rmtree(TMP_PATH)
    status_bar.progress(90)
    upload_cap.caption("Prediction...")
    st.write(text_feat)
    
    status_bar.progress(100)
    upload_cap.caption("")