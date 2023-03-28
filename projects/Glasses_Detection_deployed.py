from utils import load_lottieurl, images_dir, dlib_detector, dlib_predictor, glasses_detector_model
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from PIL import Image

def glasses_run():
    st.title("Glasses Detection")
    st.success("To detect if the subject is wearing glasses without using Neural Networks!")

    image = None

    selected = option_menu(
        menu_title="Choose an image source",
        menu_icon='camera',
        options=['View Examples', 'Upload an Image'],
        orientation='horizontal')

    if selected == 'Upload an Image':
        st.text("")
        st.subheader("Upload an image")
        k = st.file_uploader("Upload a subject's facial Image", type=['jpg','jpeg','png'])

        if k:
            st.image(k, width=150)
            image = Image.open(k).convert('RGB')
     

    else:
        st.text("")
        st.subheader("View Examples")
        with st.expander("Dropdown to use some example images"):
            col1,col2 = st.columns(2)

            with col1:
                i = Image.open(images_dir+"/Rihanna.jpeg").convert('RGB')
                i1 = i.resize((700, 750))
                st.image(i1, caption='Glasses Example')
                y = st.button("Use me", key="Glasses", use_container_width=True)

            with col2:
                j = Image.open(images_dir+"/Tom_cruise.jpeg").convert('RGB')
                j1 = j.resize((700, 750))
                st.image(j1, caption='No Glasses Example')
                n = st.button("Use me", key="No Glasses",use_container_width=True)


        if y:
            image = i

        elif n:
            image = j
        

    if image:
        if selected!='Upload an Image':
            st.image(image.resize((150, 200)))
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
            
        if output == 'No face detected':
            st.error("No Face Detected! Try another image with a persons face!")
