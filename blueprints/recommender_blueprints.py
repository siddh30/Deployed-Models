import streamlit as st

#### ADDING MAIN PROJECT FUNCTIONS AND SUB FUNCTIONS INCASE I ADD MORE SIMILAR PROJECTS

#### I. Statistical Image Classification

def recom_blueprints():
    st.title("Restaurant Recommendation Blueprints")
    st.success("This flowchart illustrates the pipeline for Twitter Sentiment Analysis.")
    st.image("images/restaurant_recommendations.png")
    with st.expander("Explaination"):
        st.markdown(''' This project is from a standalone streamlit app : https://siddh30-fastfoodie-a-restaurant-recommendation--homepage-tde7ec.streamlit.app/
        
- Data Collection - The Data needed for this app was scraped from the TripAdvisor Website using a multilayered web scraping algorithm scarping features from from 20 different in Ney Jersey, New York, California, Texas & Washington.

- Backend Programming - The entire backend programming has been done in python, using pandas, numpy, etc.

- Cleaning and Organizing the data.

- Building a Recommender System based on Weighted Content Based Filtering.
 ''')

       