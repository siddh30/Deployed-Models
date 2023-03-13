from utils import load_lottieurl, model_sentence_transformer, classifier

import streamlit as st
from streamlit_lottie import st_lottie
from sentence_transformers import SentenceTransformer
import pandas as pd
from xgboost import XGBClassifier

def twitter_run():
    st.title("Twitter Sentiment Analysis")

    st.text("")
    text = st.text_area("Tweet Something!")

    model = model_sentence_transformer
    


    if text:
        embeddings = pd.DataFrame(model.encode([text]))
        output  = classifier.predict(embeddings)


        if output==1:
            i1 = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_ym06tzbn.json")
            st_lottie(i1, 
                    speed=1.25,
                    height=250,
                    width=250, 
                    )
            st.subheader("A negative sentiment!")


        if output==2 or output == 0:
            i1 = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_j1ivgcq2.json")
            st_lottie(i1, 
                    speed=1.25,
                    height=250,
                    width=250, 
                    )
            st.subheader("A neutral or irrelevant sentiment")

        if output==3:
            i1 = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_lrlahijx.json")
            st_lottie(i1, 
                    speed=1.25,
                    height=250,
                    width=250, 
                    )
            
            st.subheader("A positive sentiment!")


    


        
