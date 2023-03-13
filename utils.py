
import requests
import streamlit as st
import streamlit_lottie as st_lottie
import os

### load lottie images
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


images_dir = os.getcwd() + '/images'
models_dir = os.getcwd() + '/models'