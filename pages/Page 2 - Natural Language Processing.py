import streamlit as st
from projects.Twitter_Sentiment_Analysis_deployed import twitter_run
from blueprints.nlp_blueprints import nlp_blueprints
from utils import blueprints_func


l_nlp = ["Sentiment Analysis"]
nlp = st.sidebar.selectbox("Choose Project Type", l_nlp)

if nlp == l_nlp[0]:
    l_projects = ["Twitter-Sentiment-Analysis"]
    project = st.sidebar.radio("Choose a Project", options=l_projects)

    if project == l_projects[0]:
        st.sidebar.image("./images/twitter.png", width=300)
        blueprints_func(nlp_blueprints,  twitter_run, no_space=False)
        
