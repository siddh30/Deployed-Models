import streamlit as st
from projects.Twitter_Sentiment_Analysis_deployed import twitter_run

from streamlit_lottie import st_lottie
from utils import load_lottieurl

l_nlp = ["Sentiment Analysis"]
nlp = st.sidebar.selectbox("Choose Project Type", l_nlp)

if nlp == l_nlp[0]:
    l_projects = ["Twitter-Sentiment-Analysis"]
    project = st.sidebar.radio("Choose a Project", options=l_projects)
    if project == l_projects[0]:
        st.sidebar.text("")
        st.sidebar.image("./images/twitter.png", width=300)
        twitter_run()
        
