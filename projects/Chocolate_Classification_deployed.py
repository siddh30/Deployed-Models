from utils import load_lottieurl, images_dir, model_choc

import streamlit as st
from streamlit_lottie import st_lottie

import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from PIL import Image

def choc_run():
    st.title("Chocolate Classification")
    st.success("To classify between dark and white chocolate!")
    st.text("")

    st.subheader("Use these examples...")

    image = None
    col1,col2 = st.columns(2)

    with col1:
        i = Image.open(images_dir+"/Dark_choc.png").convert('RGB')
        st.image(i, width=100,caption='Dark Choclate Example')
        y = st.button("Use me", key="Dark Chocolate")

    with col2:
        j = Image.open(images_dir+"/White_choc.png").convert('RGB')
        st.image(j, width=100,caption='White Chocolate Example')
        n = st.button("Use me", key="White Chocolate")


    if y:
        image = i

    elif n:
        image = j
        

    st.text("")
    st.subheader("Or upload an Image...")

    k = st.file_uploader("Upload an image of a chocolate", type='jpg')

    if k:
        img = st.image(k, width=150)
        image = Image.open(k).convert('RGB')

    if image:
        image = image.resize((64, 64))
        image = tf.keras.utils.img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # make the prediction
        model = model_choc
        output = model.predict(image)

        if output<0.5:
                st.success("Dark chocolate!")
                i1 = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_fbr1dz4i.json")
                st_lottie(i1, 
                        speed=1.25,
                        height=250,
                        width=250, 
                        )
                
        if output>=0.5:
            st.warning("White Chocolate!")
            i1 = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_fbr1dz4i.json")
            st_lottie(i1, 
                    speed=1.25,
                    height=250,
                    width=250, 
                    )
            
