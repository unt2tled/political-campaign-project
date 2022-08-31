# Political Campaign Project
Deep learning pipelines to predict the target of political messages.
## About
The goal of this project is to present machine learning approach of classification political campaign videos from the USA of different years by target audience (base/center). The classification is done by extracting different features from the video (e.g., speech-to-text, visual data) and training a neural network. More details can be found in the related [paper]().
## Navigation
### Features extraction
All the code used for features extraction is in the */tools* directory.
### Analysis
Code for model analysis is in the */analysis* directory.
### Training model
To train the model use [Google Colab](https://colab.research.google.com/drive/1ceVEWRAkIQJsOGuMxmG2qvPY3huZf8gc?usp=sharing).
### Demo
Example UI of a pre-trained model with test accuracy of ~80% using speech-to-text and text from video features can be found [here](https://unt2tled-political-campaign-project-demo-6gbfbd.streamlitapp.com/) or by cloning the repository and calling from the root:
```
pip install streamlit
streamlit run Demo.py
```
