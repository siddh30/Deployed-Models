import streamlit as st
import sys
from utils import blueprints_func
from blueprints.recommender_blueprints import recom_blueprints
from streamlit_lottie import st_lottie
from projects.FastFoodie import fastfoodie_run
from bokeh.models.widgets import Div


l_nlp = ["Collaborative Systems"]
nlp = st.sidebar.selectbox("Choose Project Type", l_nlp)

if nlp == l_nlp[0]:
    l_projects = ['FastFoodie - A web app']
    project = st.sidebar.radio("Choose a Project", options=l_projects)

    if project == l_projects[0]:

        st.sidebar.image("https://github.com/siddh30/FastFoodie-A-Restaurant-Recommendation-App/blob/main/Data/App_icon.png?raw=true", width=300)
        blueprints_func(recom_blueprints, fastfoodie_run)
