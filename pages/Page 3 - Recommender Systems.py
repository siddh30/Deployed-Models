import streamlit as st
import sys
from utils import load_lottieurl
from streamlit_lottie import st_lottie
from projects.FastFoodie import fastfoodie_run
from bokeh.models.widgets import Div


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
        st.success("Applying Collaborative Filtering to get the most similar restaurants from TripAdvisor.")
       
        url = "https://siddh30-fastfoodie-a-restaurant-recommendation--homepage-tde7ec.streamlit.app/"
        if st.button("Link to Standalone App!", use_container_width=True):
                #js = "window.open" + "('" + url + "')"  # New tab
                js = "window.location.href" + " = " + "'" + url + "'"  # Current tab
                html = '<img src onerror="{}">'.format(js)
                div = Div(text=html)
                st.bokeh_chart(div)


        fastfoodie_run()

        # i1 = load_lottieurl("https://assets9.lottiefiles.com/private_files/lf30_UlXgnV.json")
        # st_lottie(i1, 
        #         speed=1.00,
        #         height=500,
        #         width=500, 
        #         )


# if nlp == l_nlp[1]:
#     l_projects = ['Spotify Music Recommendation']
#     project = st.sidebar.radio("Choose a Project", options=l_projects)

#     if project == l_projects[0]:
#         st.sidebar.text("")
#         st.sidebar.image("https://www.freepnglogos.com/uploads/spotify-logo-png/spotify-download-logo-30.png", width=300)
#         st.title('Spotify Music Recommendation')
    
