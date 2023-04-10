import streamlit as st

#### ADDING MAIN PROJECT FUNCTIONS AND SUB FUNCTIONS INCASE I ADD MORE SIMILAR PROJECTS

#### I. Statistical Image Classification

def nlp_blueprints():
    st.title("Twitter Sentiment Blueprints")
    st.success("This flowchart illustrates the pipeline for Twitter Sentiment Analysis.")
    st.image("images/sentiment.png")
    with st.expander("Explaination"):
        st.markdown('''In this flow we take tweets for a text database and pass it thourgh a pretrained Sentence Transformer.
        from HuggingFace. This generates a 768 column vector (Context Vector) embeddings. These embeddings are then fed to an XgBoost Classifer
        with labeled Sentiments as part of training. ''')

       