import streamlit as st
import tools.ocr_video as ocr
import os
import shutil
import speech_recognition as sr 
import moviepy.editor as mp
from pydub import AudioSegment
from pydub.silence import split_on_silence
# Temporary folder path
TMP_PATH = "tmp/"

r = sr.Recognizer()

def get_large_audio_transcription(path):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    # open the audio file using pydub
    sound = AudioSegment.from_wav(path)  
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                whole_text += text
    # return the text for all chunks detected
    return whole_text

st.title("Demo page")
st.markdown("""Upload the US political campaign video to predict its orientation (base/center). 
         To train the model go to "**Training**" section.""")
video_file = st.file_uploader("Choose the US political campaign video", type=["wmv", "avi", "mov"])
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
    # Speech-to-text
    upload_cap.caption("Extracting speech from audio...")
    clip = mp.VideoFileClip(TMP_PATH+"uploaded_video_tmp")
    clip.audio.write_audiofile(TMP_PATH+"uploaded_audio_tmp.wav")
    speech_text = get_large_audio_transcription(TMP_PATH+"uploaded_audio_tmp.wav")
    text_feat = speech_text + " " + text_feat
    clip.close()
    shutil.rmtree(TMP_PATH)
    status_bar.progress(90)
    upload_cap.caption("Prediction...")
    st.write(text_feat)
    
    status_bar.progress(100)
    upload_cap.caption("")