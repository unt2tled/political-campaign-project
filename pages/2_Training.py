import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from training.trainer import CampaignTextModel
from training.parser import TagFilesGenerator

LABELS_MAP = {0: "center", 1: "both", 2: "base"}

def name_filter_by_head(name: str,head_size: int,head_content: str) -> bool:
    return head_size<=0 or name[:head_size] == head_content
    
def non_both_data(x):
    return x['label']!= 1

st.header("Training")
st.markdown("""The model is trained on hundreds of the US political campaign videos based on different features (speech-to-text, faces sentiment, OCR, etc.).
            This page demonstrates the model training (the trained model will be used on "**Demo**" page).""")
# Add options
tagging_path = st.text_input("Labeled data path", "training/data/tagging_db.csv")
df = pd.read_csv(tagging_path)
st.dataframe(df)
epochs_num = st.slider("Number of epochs", 1, 40, 5)
batch_size = st.slider("Batch size", 1, 100, 30)
splits_num = st.slider("Number of splits for speech-to-text", 1, 100, 1)
test_size = st.slider("Test size (%)", 1, 100, 40)
lang_model_name = st.text_input("Language model name", "distilbert-base-uncased")
# Add training button
if st.button("Train Model"):
    # Trainer options
    PATH_NAME = tagging_path
    FILTER_DEFINITE = True
    SPLIT_NUM=splits_num
    SHORTEN_CENTER_TO_BASE_SIZE=True
    PRIOATIZE_AMERICANS=False
    FILTER_FUNC = None
    # Generate tags file
    tfg = TagFilesGenerator(PATH_NAME,FILTER_DEFINITE,SPLIT_NUM,SHORTEN_CENTER_TO_BASE_SIZE,PRIOATIZE_AMERICANS,
                        FILTER_FUNC)
    tfg.run()
    LANG_MODEL_NAME = lang_model_name
    DIRECTION = None
    PATH_NAME = "tags.csv"
    TEST_SIZE = test_size / 100
    DATA_FILTER_FUNC = None
    TEST_FILTER_FUNC = None
    m = CampaignTextModel(LANG_MODEL_NAME, DIRECTION, PATH_NAME, TEST_SIZE, DATA_FILTER_FUNC, TEST_FILTER_FUNC)
    m.load_data_from_csv()
    m.dataset = m.dataset.shuffle(seed = None)
    m.train(epochs=epochs_num, per_device_train_batch_size=batch_size, per_device_eval_batch_size=39, early_stopping_patience=3)
    # Save model
    m.trainer.save_model("trained_model")
    # Print success message
    st.success("The model was successfully trained.")
    # Print logs
    st.subheader("Test dataset")
    st.markdown("Train dataset size: **"+str(len(m.dataset["train"]))+"**.")
    st.markdown("Test dataset size: **"+str(len(m.dataset["test"]))+"**.")
    test_pred = m.predict_by_dataset(m.dataset["test"])
    true_labels = m.dataset["test"]["label"]
    num_correct = np.sum([test_pred[i] == true_labels[i] for i in range(len(test_pred))])
    test_accuracy = num_correct/len(test_pred) * 100
    conf_mat = confusion_matrix(true_labels,test_pred,labels=list(range(2 if m.direction != None else 3)))
    test_df = pd.DataFrame(data=m.dataset["test"])
    test_df["predicted_labels"] = test_pred
    test_df.rename(columns={"label": "true_label"}, inplace = True)
    test_df.replace({"true_label": LABELS_MAP, "predicted_labels": LABELS_MAP}, inplace=True)
    st.dataframe(test_df)
    st.markdown("Test accuracy: **"+str(test_accuracy)+"**%.")
    st.subheader("Confusion martix")
    test_df=pd.DataFrame(confusion_matrix(true_labels, test_pred, labels=list(range(2 if m.direction != None else 3))), ["center", "both",  "base"], ["center", "both",  "base"])
    st.dataframe(test_df)