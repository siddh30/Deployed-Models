from utils import load_lottieurl, images_dir, dlib_detector, dlib_predictor, glasses_detector_model
import streamlit as st
from streamlit_lottie import st_lottie
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import dlib
import cv2
from PIL import Image

def glasses_run():
    st.title("Glasses Detection")
    st.success("To detect if the subject is wearing glasses without using Neural Networks!")
    st.text("")

    st.subheader("Use these examples...")

    image = None
    col1,col2 = st.columns(2)

    with col1:
        i = Image.open(images_dir+"/Rihanna.jpeg").convert('RGB')
        i1 = i.resize((120, 150))
        st.image(i1, caption='Glasses Example')
        y = st.button("Use me", key="Glasses")

    with col2:
        j = Image.open(images_dir+"/Tom_cruise.jpeg").convert('RGB')
        j1 = j.resize((120, 150))
        st.image(j1, caption='No Glasses Example')
        n = st.button("Use me", key="No Glasses")


    if y:
        image = i

    elif n:
        image = j
        

    st.text("")
    st.subheader("Or upload an Image...")

    k = st.file_uploader("Upload a subject's facial Image", type=['jpg','jpeg','png'])

    if k:
        st.image(k, width=150)
    

    if image:
        ### make the prediction
        output = glasses_detector_model(image, detector=dlib_detector, predictor=dlib_predictor)
        
        if output==1:
            st.success("Glasses Detected!")
            i1 = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_AnTU82FUFQ.json")
            st_lottie(i1, 
                    speed=1.25,
                    height=250,
                    width=250, 
                    )
                        
        if output==0:
            st.error("No Glasses Detected!")
            i1 = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_ieemc0fs.json")
            st_lottie(i1, 
                    speed=1.25,
                    height=250,
                    width=250, 
                            )