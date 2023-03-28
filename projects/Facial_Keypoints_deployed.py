from utils import load_lottieurl, images_dir, FaceDetection, faces_oneChannel_96, keypoints_torch_model
import streamlit as st
from streamlit_lottie import st_lottie
import numpy as np
from streamlit_option_menu import option_menu
from PIL import Image

def keypoints_run():
    st.title("Facial Keypoints Generation")
    st.success("To generate keypoints given a subjects face!")
    st.text("")

    image = None

    selected = option_menu(
        menu_title="Choose an image source",
        menu_icon='camera',
        options=['View Examples', 'Upload an Image'],
        orientation='horizontal')

    if selected == 'Upload an Image':
        st.text("")
        st.subheader("Upload an Image...")
        k = st.file_uploader("Upload an image containing faces", type=['jpg','jpeg','png'])

        if k:
            img = st.image(k, width=150)
            image = Image.open(k).convert('RGB')

        
    else:
        st.text("")
        st.subheader("View Examples")
        with st.expander("Dropdown to use some example images"):
            i = Image.open(images_dir+"/messi_ronaldo.jpg").convert('RGB')
            i1 = i.resize((750, 450))
            st.image(i1)
            y = st.button("Use me", key="Glasses", use_container_width=True)

            image = None

            if y:
                image = i

    if image:
        face_detection = FaceDetection(np.array(image))
        st.warning("Step 1: Face Detection")
        faces_crop = face_detection.detection()

        if faces_crop == []:
            st.error("No Face Detected! Try another image with a persons face!")

        else:
            face_detection.number_faces()

            st.warning("Step 2: Facial Keypoints on Cropped Images")
            faces_oneChannel_96(faces_crop,model=keypoints_torch_model)

            st.success("Scan Complete!")



