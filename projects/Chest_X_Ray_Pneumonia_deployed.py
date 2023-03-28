
from utils import load_lottieurl, images_dir, model_chest
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import tensorflow as tf 
from PIL import Image

def chest_xray_run():
    st.title("Chest X-Ray Pneuomonia Detection")
    st.success("To detect signs of Pneuomonia by looking at a subject's Chest X-Ray!")
    image = None

    selected = option_menu(
        menu_title="Choose an image source",
        menu_icon='camera',
        options=['Use Given Examples', 'Upload an Image'],
        orientation='horizontal')

    if selected == 'Upload an Image':
        st.text("")
        st.subheader("Upload an image")
        k = st.file_uploader("Upload a Chest X-ray Image", type=['jpg','jpeg','png'])

        if k:
            st.image(k, width=150)
            image = Image.open(k).convert('RGB')

    else:
        st.subheader("Use given examples")
        with st.expander("Dropdown to use some example images"):
            st.subheader("Use these examples...")

            col1,col2 = st.columns(2)

            with col1:
                i = Image.open(images_dir+"/Pneumonia.jpg").convert('RGB').resize((700, 750))
                st.image(i, caption='Pnuemonia Example')
                y = st.button("Use me", key="Pnuemonia", use_container_width=True)

            with col2:
                j = Image.open(images_dir+"/Clear.jpg").convert('RGB').resize((700, 750))
                st.image(j, caption='No Pnuemonia Example')
                n = st.button("Use me", key="No Pnuemonia", use_container_width=True)


            if y:
                image = i

            elif n:
                image = j     

    if image:
        if selected!='Upload an Image':
            st.image(image.resize((150, 200)))
        image = image.resize((64, 64))
        image = tf.keras.utils.img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # make the prediction

        
        model = model_chest
        output = model.predict(image)

        if output<0.5:
                st.success("No Pnuemonia Detected!")
                i1 = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_C4oUGiFMYZ.json")
                st_lottie(i1, 
                        speed=1.25,
                        height=250,
                        width=250, 
                        )
                
        if output>=0.5:
            st.error("Pnuemonia Detected!")
            i1 = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_p7ki6kij.json")
            st_lottie(i1, 
                    speed=1.25,
                    height=250,
                    width=250, 
                    )
            
