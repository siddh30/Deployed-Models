import streamlit as st
from streamlit_lottie import st_lottie
from utils import load_lottieurl
from bokeh.models.widgets import Div
import urllib

st.sidebar.image('images/me.png', width=300, caption="AI/ML Data Scientist")

st.title("About Me")

st.success('My name is Siddharth Mandgi and I am a Data scientist applying principles of AI/ML & Data Science to help organizations tackle challenges, automate workflows and build profitable solutions in People Analytics.I love working on projects involving Natural Language Processing, Computer Vision, Recommender Systems, and Python. I am a Kaggle Expert and a novice blogger on Medium. I love gaming, dining, coding, playing soccer, and making & mixing music which I occasionally upload on SoundCloud.')


# with st.expander("My Socials!"):

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         url = "https://www.linkedin.com/in/siddharthmandgi/"
#         if st.button("Linkedin", use_container_width=True):
#                 js = "window.open" + "('" + url + "')"  # New tab
#                 html = '<img src onerror="{}">'.format(js)
#                 div = Div(text=html)
#                 st.bokeh_chart(div)

#     with col2:
#         url = "https://github.com/siddh30"
#         if st.button("Github", use_container_width=True):
#                 js = "window.open" + "('" + url + "')"  # New tab
#                 html = '<img src onerror="{}">'.format(js)
#                 div = Div(text=html)
#                 st.bokeh_chart(div)

#     with col3:
#         url = "https://medium.com/@siddh30"
#         if st.button("Medium", use_container_width=True):
#                 js = "window.open" + "('" + url + "')"  # New tab
#                 html = '<img src onerror="{}">'.format(js)
#                 div = Div(text=html)
#                 st.bokeh_chart(div)

        
