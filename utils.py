import os
import requests
import streamlit as st
import streamlit_lottie as st_lottie
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

### load lottie images
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


images_dir = os.getcwd() + '/images'
models_dir = os.getcwd() + '/models'


@st.cache_resource
def load_computer_vision_model(path):
    model = tf.keras.models.load_model(path)
    return model

model_chest = load_computer_vision_model(models_dir+'/chest_xray_saved_models/bestmodel_0')
model_choc = load_computer_vision_model(models_dir+'/Chocolate_saved_models/bestmodel')


@st.cache_resource
def load_nlp_model(path_model, path_xgboost_head):
    model_sentence_transformer = SentenceTransformer(path_model)
    classifier = XGBClassifier()
    classifier.load_model(path_xgboost_head)
    return model_sentence_transformer, classifier

model_sentence_transformer, classifier = load_nlp_model(path_model='paraphrase-mpnet-base-v2', path_xgboost_head=models_dir+"/twitter_sentiment_xgb_model.json")