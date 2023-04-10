import streamlit as st
import sys
sys.path.append("/Users/siddharthmandgi/Desktop/Data-Science-Universe/Projects/Deployed_models/")
from projects import Chest_X_Ray_Pneumonia_deployed as chst, Chocolate_Classification_deployed as choc, Glasses_Detection_deployed as glass, Facial_Keypoints_deployed as keypoints
import tensorflow as tf
from streamlit_lottie import st_lottie
from utils import blueprints_func
from blueprints.computer_vision_blueprints import pnuemonia_blueprint, chocolate_blueprint, glasses_blueprints, keypoints_blueprints

###### PAGE FOR ALL COMPUTER VISION PROJECTS

l_cv = ["Statistical Image Classification", "Neural Image Classification", "Image Attribute Generation"]
project_cv = st.sidebar.selectbox("Choose Project Type", l_cv)

if  project_cv == l_cv[0]:
    l_projects = ["Glasses Detection",]
    project = st.sidebar.radio("Choose a Project", options=l_projects)
    
    if project == l_projects[0]:
        st.sidebar.image("./images/cv.png", width=300)
        blueprints_func(glasses_blueprints,  glass.glasses_run)


if  project_cv == l_cv[1]:
    l_projects = ["Chest X-ray Pnuemonia Detection", 'Chocolate Classification']
    project = st.sidebar.radio("Choose a Project", options=l_projects)
    
    
    if project == l_projects[0]:
        st.sidebar.image("./images/cv.png", width=300)
        blueprints_func(pnuemonia_blueprint, chst.chest_xray_run)


    if project == l_projects[1]:
        st.sidebar.image("./images/cv.png", width=300)
        blueprints_func(chocolate_blueprint, choc.choc_run)
        


if  project_cv == l_cv[2]:
    l_projects = ["Facial Keypoints Generation"]
    project = st.sidebar.radio("Choose a Project", options=l_projects)
    
    if project == l_projects[0]:
        st.sidebar.image("./images/cv.png", width=300)
        blueprints_func(keypoints_blueprints, keypoints.keypoints_run)


