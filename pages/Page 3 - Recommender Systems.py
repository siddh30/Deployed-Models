import streamlit as st
import sys
from utils import load_lottieurl
from streamlit_lottie import st_lottie
sys.path.append("/Users/siddharthmandgi/Desktop/Data-Science-Universe/Projects/Deployed_models_streamlit/")


# l_nlp = ["Content-Based Systems", "Collaborative Systems", ]
l_nlp = ["Collaborative Systems"]
nlp = st.sidebar.selectbox("Choose Project Type", l_nlp)

if nlp == l_nlp[0]:
    l_projects = ['FastFoodie - A web app']
    project = st.sidebar.radio("Choose a Project", options=l_projects)

    if project == l_projects[0]:
        st.sidebar.text("")
        st.sidebar.image("https://github.com/siddh30/FastFoodie-A-Restaurant-Recommendation-App/blob/main/Data/App_icon.png?raw=true", width=300)
        st.title('FastFoodie - A Resturant Recommendation App')
        st.success("I have a deployed another streamlit app which covers restaurants form 20 cities across New York, New Jersey, California, Texas and Washington to recommend the 10 most similar restaurants to the one you like.This app uses Natural Language Processing and Content Based Recommender Systems with focusing on user comments as the main feature.")
        link = '[Link to the app!](https://siddh30-fastfoodie-a-restaurant-recommendation--homepage-tde7ec.streamlit.app/)'
        st.markdown(link, unsafe_allow_html=True)

        i1 = load_lottieurl("https://assets9.lottiefiles.com/private_files/lf30_UlXgnV.json")
        st_lottie(i1, 
                speed=1.00,
                height=500,
                width=500, 
                )


# if nlp == l_nlp[1]:
#     l_projects = ['Spotify Music Recommendation']
#     project = st.sidebar.radio("Choose a Project", options=l_projects)

#     if project == l_projects[0]:
#         st.sidebar.text("")
#         st.sidebar.image("https://www.freepnglogos.com/uploads/spotify-logo-png/spotify-download-logo-30.png", width=300)
#         st.title('Spotify Music Recommendation')
    
