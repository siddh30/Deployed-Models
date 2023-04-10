import streamlit as st

#### ADDING MAIN PROJECT FUNCTIONS AND SUB FUNCTIONS INCASE I ADD MORE SIMILAR PROJECTS

#### I. Statistical Image Classification

def glasses_blueprints():
    st.title("Glasses Detection Blueprints")
    st.success("This flowchart illustrates the pipeline for Glasses Detection.")
    st.image("images/glasses_detection.png")
    with st.expander("Explaination"):
        st.markdown('''In this flow chart all images for Glasses Detection is sources from an Image Database
                        and passed through a DLIB Face Detector. The purpose of this is to ensure we avoid misclassifications
                        of arbitary image uploads when passed through Canny Filters.''')

        st.subheader("Stages of Detection")
        col1, col2, col3 = st.columns(3)
    

        with col1:
            st.success("1. Take an Input Image")
            st.image("images/Rihanna.jpeg", use_column_width=True)

        with col2:
            st.success("2. Detect Keypoints and Crop to Nose Bridge using DLIB")
            st.image("images/cropped.png", use_column_width=True)

        with col3:
            st.success("3. Get the Edges and Classify")
            st.image("images/canny_filters_edges.png", use_column_width=True)

#### II. Neural Classification
def Neural_Image():
    st.success("This flowchart illustrates the pipeline for projects under Neural Image Classification.")
    st.image("images/Neural_Image.png")
    with st.expander("Explaination"):
        st.markdown('''In this flow chart all images for Neural Image Classifictaion is sourced from an Image Database
                        and passed through a Multiclass Image classifier. The purpose of this is to ensure we avoid misclassifications
                        of arbitary image uploads by secondary classifiers.''')

def pnuemonia_blueprint():
    st.title("Pnuemonia Classifiction Blueprints")
    Neural_Image()

    st.text(" ")
    st.markdown("Focusing on Pnuemonia Classification Model")
    st.success('''I use ImageData Generator where I rescale the image by 255, and assign 
    zoom range, rotation range, width shift_range, height range = 0.1. I am using a Tensorflow Network 
    to build a model and classify.Keras is just an API of Tensorflow 2.0 to make tf highly approachable 
    and easy for model buidling.I apply Binary Cross Entropy Loss, train it for 25 epochs and callback 
    on saving the best model monitoring validation loss. ''')

    with st.expander("Model Architecture"):
        st.image("images/chest_arch.png")


def chocolate_blueprint():
    st.title("Chocolate Classifiction Blueprints")
    Neural_Image()

    st.text(" ")
    st.markdown("Focusing on Chocolate Classification Model")
    st.success('''I am using a Tensorflow Network to build a model and classify. Keras is just an 
    API of Tensorflow 2.0 to make tf highly approachable and easy for model buidling.
    I apply Binary Cross Entropy Loss, train it for 25 epochs and callback on saving the 
    best model monitoring validation loss. ''')

    with st.expander("Model Architecture"):
        st.image("images/choc_arch.png")


#### III. Image Attribute Generation
def keypoints_blueprints():
    st.title("Facial KeyPoints Blueprints")
    st.success(''' In this project we have addressed the problem proposed above by creating a model that can detect the facial features from the image dataset. 
    The main goal is to obtain the coordinates of eyes, eyebrows, nose and mouth in the picture. 
    These coordinates are known as keypoints.''')
    

    st.markdown(''' In order to be more specific about the location and orientation of these keypoints, 
    it will be necessary in some cases to assign more than one keypoint for each facial feature. 
    This way, the face of the subject can be perfectly defined. For this dataset, our model will provide the following keypoints:

    - Eyes: For both eyes the model will predict three coordinates corresponding to the center, inner and outer parts of the eyes.

    - Eyebrows: For this feature the model will yield two coordinates corresponding to the inner and outer side for both of the eyebrows.

    - Nose: In this case, one coordinate will be enough.

    - Mouth: For the last feature, the model will give four coordinates, corresponding to the left, right, top and bottom part of the lip. 
    This way the computer could actually read the mood of the subject. ''')

    st.success('''In the past few years, advancements in Facial Key Points detection have been made by the application of Deep Convolutional Neural Network (DCNN). 
    DCNNs have helped build state-of-the-art models for image recognition, recommender systems, natural language processing, etc. 
    Our intention is to make use of these architectures to address this problem, trying to use different algorithms to study which are more suitable for the proposed task''')

    with st.expander("Model Architecture"):
        st.image("images/CNN_Facial_Keypoints.png")
