# Political Campaign Project
Deep learning pipelines to predict the target of political messages.
## About
The goal of this project is to present machine learning approach of classification political campaign videos from the USA of different years by target audience (base/center). The classification is done by extracting different features from the video (e.g., speech-to-text, visual data) and training a neural network. More details can be found in the related [paper](https://drive.google.com/drive/folders/1-7rkd_SozNGLrNHXnEZ0iTKqO9ztKhiU).
## Navigation
### Dataset
Datasets, including extracted features, tagging files and political campaign videos to train on can be found [here](https://drive.google.com/drive/folders/1-7rkd_SozNGLrNHXnEZ0iTKqO9ztKhiU?usp=sharing).
### Features extraction
All the code used for features extraction is in the */tools* directory.
### Analysis
Code for model analysis is in the */analysis* directory. [This](https://colab.research.google.com/drive/19RLpj0W5k1WzRj0UOJO9J-vq_F-kbAkU?usp=sharing) Google Colab notebook is used for face sentiment and color analysis from videos.
### Training model
To train the model use [this](https://colab.research.google.com/drive/1ceVEWRAkIQJsOGuMxmG2qvPY3huZf8gc?usp=sharing) Google Colab notebook. [This](https://colab.research.google.com/drive/1MH19zWCCqQFTKidT5qq6pIPbmsdyuAIp?usp=sharing) notebook is used to make predictions from the pre-trained model.
### Demo
Example UI of a pre-trained model with test accuracy of ~80% using speech-to-text and text from video features can be found [here](https://huggingface.co/spaces/Vanofuture/political_campaign) or by cloning the repository and calling from the project's root:
```
pip install streamlit
streamlit run Demo.py
```
