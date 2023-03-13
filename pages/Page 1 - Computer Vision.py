import streamlit as st
import sys
sys.path.append("/Users/siddharthmandgi/Desktop/Data-Science-Universe/Projects/Deployed_models/")
from projects import Chest_X_Ray_Pneumonia_deployed as chst, Chocolate_Classification_deployed as choc

from streamlit_lottie import st_lottie
from utils import load_lottieurl

with st.sidebar:
    l_nlp = ["Image Classification"]
    nlp = st.selectbox("Choose Project Type", l_nlp)

    if nlp == l_nlp[0]:
        l_projects = ["Chest X-ray Pnuemonia Detection", 'Chocolate Classification']
        project = st.radio("Choose a Project", options=l_projects)

    st.image("https://www.pngkit.com/png/full/88-880499_platform-02-computer-vision-icon.png", width=300)
        
if project == l_projects[0]:
    chst.chest_xray_run()

if project == l_projects[1]:
    choc.choc_run()