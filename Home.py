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

st.sidebar.success(" I am always building new projects so look out more for on the way!")

with st.sidebar.expander("Upcoming Updates!"):
    st.warning("1. Adding Surrogate Image Classification Models to avoid users from misclassifying on incorrect image uploads.")
    st.warning("2. Adding Blueprints Page for each project explaining model flows.")
