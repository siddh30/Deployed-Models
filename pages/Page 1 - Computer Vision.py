import streamlit as st
import sys
sys.path.append("/Users/siddharthmandgi/Desktop/Data-Science-Universe/Projects/Deployed_models/")
from projects import Chest_X_Ray_Pneumonia_deployed as chst, Chocolate_Classification_deployed as choc, Glasses_Detection_deployed as glass, Facial_Keypoints_deployed as keypoints
import tensorflow as tf
from streamlit_lottie import st_lottie
from utils import load_lottieurl, models_dir, images_dir

l_cv = ["Statistical Image Classification", "Neural Image Classification", "Image Attribute Generation"]
project_cv = st.sidebar.selectbox("Choose Project Type", l_cv)

if  project_cv == l_cv[0]:
    l_projects = ["Glasses Detection",]
    project = st.sidebar.radio("Choose a Project", options=l_projects)
    
    if project == l_projects[0]:
        glass.glasses_run()


if  project_cv == l_cv[1]:
    l_projects = ["Chest X-ray Pnuemonia Detection", 'Chocolate Classification']
    project = st.sidebar.radio("Choose a Project", options=l_projects)
    
    if project == l_projects[0]:
        chst.chest_xray_run()

    if project == l_projects[1]:
        choc.choc_run()


if  project_cv == l_cv[2]:
    l_projects = ["Facial Keypoints Generation"]
    project = st.sidebar.radio("Choose a Project", options=l_projects)
    
    if project == l_projects[0]:
        keypoints.keypoints_run()


st.sidebar.image("./images/cv.png", width=300)