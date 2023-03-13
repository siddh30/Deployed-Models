import streamlit as st
from streamlit_lottie import st_lottie
from utils import load_lottieurl

st.sidebar.image('images/me.png', width=300, caption="AI/ML Data Scientist")

st.title("About Me")

st.success('Data scientist applying principles of AI/ML & Data Science to help the firm tackle challenges, automate workflows and build profitable solutions in People Analytics.I love working on projects involving Natural Language Processing, Computer Vision, Recommender Systems, and Python. I am a Kaggle Expert and a novice blogger on Medium. I love gaming, dining, coding, playing soccer, and making & mixing music which I occasionally upload on SoundCloud.')

col1, col2, col3 = st.columns(3)


with col1:
    link = '[Linkedin](https://www.linkedin.com/in/siddharthmandgi/)'
    st.markdown(link, unsafe_allow_html=True)

with col2:
    link = '[Github](https://github.com/siddh30)'
    st.markdown(link, unsafe_allow_html=True)

with col3:
    link = '[Medium](https://medium.com/@siddh30)'
    st.markdown(link, unsafe_allow_html=True)

    
