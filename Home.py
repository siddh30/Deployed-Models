import streamlit as st
from streamlit_lottie import st_lottie
from utils import load_lottieurl



st.title("Deployed Projects Porftolio")
st.text("A one-stop-shop for deployed github projects.")


i1 = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_FBIFgdSAN8.json")

st_lottie(i1, 
          speed=1.25,
          height=500,
          width=500,
          key="hello")

st.text("Click on the arrow icon on the top left to navigate to different projects...")

st.sidebar.success(" I am always building new projects so look out more for on the way!")