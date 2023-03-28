from utils import load_lottieurl, images_dir, model_choc

import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import tensorflow as tf 
from PIL import Image

def choc_run():
    st.title("Chocolate Classification")
    st.success("To classify between dark and white chocolate!")
    st.text("")

    st.subheader("Use these examples...")

    image = None
    
    selected = option_menu(
        menu_title="Choose an image source",
        menu_icon='camera',
        options=['Use Given Examples', 'Upload an Image'],
        orientation='horizontal')

    if selected == 'Upload an Image':
        st.text("")
        st.subheader("Upload an image")

        k = st.file_uploader("Upload an image of a chocolate", type=['jpg','jpeg','png'])

        if k:
            st.image(k, width=150)
            image = Image.open(k).convert('RGB')


    else:
        st.subheader("Use given examples")
        with st.expander("Dropdown to use some example images"):
            col1,col2 = st.columns(2)

            with col1:
                i = Image.open(images_dir+"/Dark_choc.png").convert('RGB').resize((700, 750))
                st.image(i, caption='Dark Choclate Example')
                y = st.button("Use me", key="Dark Chocolate", use_container_width=True)

            with col2:
                j = Image.open(images_dir+"/White_choc.png").convert('RGB').convert('RGB').resize((700, 750))
                st.image(j, caption='White Chocolate Example')
                n = st.button("Use me", key="White Chocolate", use_container_width=True)


            if y:
                image = i

            elif n:
                image = j
            

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
            
