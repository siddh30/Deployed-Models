from utils import load_lottieurl, images_dir, FaceDetection, faces_oneChannel_96, keypoints_torch_model
import streamlit as st
from streamlit_lottie import st_lottie
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def keypoints_run():
    st.title("Facial Keypoints Generation")
    st.success("To generate keypoints given a subjects face!")
    st.text("")

    st.subheader("Use this example...")

    i = Image.open(images_dir+"/messi_ronaldo.jpg").convert('RGB')
    i1 = i.resize((250, 150))
    st.image(i1)
    y = st.button("Use me", key="Glasses")

    image = None

    if y:
        image = i

    
    st.text("")
    st.subheader("Or upload an Image...")

    k = st.file_uploader("Upload an image containing faces", type=['jpg','jpeg','png'])

    if k:
        img = st.image(k, width=150)
        image = Image.open(k).convert('RGB')

        
    if image:
        face_detection = FaceDetection(np.array(image))
        st.warning("Step 1: Face Detection")
        faces_crop = face_detection.detection()
        face_detection.number_faces()

        st.warning("Step 2: Facial Keypoints on Cropped Images")
        faces_oneChannel_96(faces_crop,model=keypoints_torch_model)

        st.success("Scan Complete!")


    else:
        st.warning("Begin scanning - Upload an image or Use the above example ")
        i1 = load_lottieurl("https://assets4.lottiefiles.com/temp/lf20_XcJCfR.json")
        st_lottie(i1, 
                speed=1.25,
                height=250,
                width=250, 
                )

