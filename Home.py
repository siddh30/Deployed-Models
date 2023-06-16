import streamlit as st
from streamlit_lottie import st_lottie
from utils import load_lottieurl



st.title("Deployed Projects Porftolio")
st.text("A one-stop-shop for deployed github projects.")


i1 = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_FBIFgdSAN8.json")

st_lottie(i1, 
          speed=1.25,
          height=350,
          width=350,
          key="hello")

st.success("Click on the arrow icon on the top left to navigate to different projects...")


with st.sidebar:
    st.markdown(f"""
               Winner - #Streamlit25k Challenge \n""")
    st.image("images/winner.png")
    st.markdown(f"""
               Link to Post - https://www.linkedin.com/feed/update/urn:li:activity:7075587186087059456/
               """)