import os
import requests
import streamlit as st
import streamlit_lottie as st_lottie
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from PIL import Image

### load lottie images
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


images_dir = os.getcwd() + '/images'
models_dir = os.getcwd() + '/models'



#### COMPUTER VISION

##### 1. Neural Classification Model
@st.cache_resource
def load_computer_vision_model(path):
    model = tf.keras.models.load_model(path)
    return model

model_chest = load_computer_vision_model(models_dir+'/chest_xray_saved_models/bestmodel_0')
model_choc = load_computer_vision_model(models_dir+'/Chocolate_saved_models/bestmodel')

##### 2. Extra Detectors
@st.cache_resource
def extra_detectors(haarcascade_detector_path, shape_predictors_path):
    #haarcascade
    face_cascade = cv2.CascadeClassifier(haarcascade_detector_path)

    #dlib
    dlib_detector = dlib.get_frontal_face_detector()
    dlib_predictor = dlib.shape_predictor(shape_predictors_path) 

    return face_cascade, dlib_detector, dlib_predictor
haarcascade_model, dlib_detector, dlib_predictor = extra_detectors(haarcascade_detector_path=models_dir+"/haarcascade_frontalface_default.xml", 
                                                                  shape_predictors_path=models_dir+"/shape_predictor_68_face_landmarks.dat")



### 3. Glasses model
def glasses_detector_model(image, detector, predictor):
    img = np.array(image)
    if len(detector(img))==0:
        return('No face detected')
    rect = detector(img)[0]
    sp = predictor(img, rect)
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])

    nose_bridge_x = []
    nose_bridge_y = []

    for i in [28,29,30,31,33,34,35]:
        nose_bridge_x.append(landmarks[i][0])
        nose_bridge_y.append(landmarks[i][1])

    ### x_min and x_max
    x_min = min(nose_bridge_x)
    x_max = max(nose_bridge_x)

    ### ymin (from top eyebrow coordinate),  ymax
    y_min = landmarks[20][1]
    y_max = landmarks[29][1]

    img2 = image
    img2 = img2.crop((x_min,y_min,x_max,y_max))

    img_blur = cv2.GaussianBlur(np.array(img2),(3,3), sigmaX=0, sigmaY=0)

    edges = cv2.Canny(image =img_blur, threshold1=100, threshold2=200)

    edges_center = edges.T[(int(len(edges.T)/2))]
    
#     plt.imshow(edges, cmap =plt.get_cmap('gray'))

    if 255 in edges_center:
        return(1)
    else:
        return(0)


### 4. Face Detection and landmarks model 
class FaceDetection():
    
    def __init__(self, img, path2class = 'models/haarcascade_frontalface_default.xml'):
        
        #Load image
        self.img_original = img

        # Convert to RGB colorspace
        self.img_original = self.convertToRGB(self.img_original)
        
        # copy original image
        self.img_with_detections = np.copy(self.img_original)
        
        #convert image to gray (opencv expects gray images)
        self.gray_img = self.convertToGray(self.img_original)

        #load cascade classifier (haarcascade) training file
        self.haar_face_cascade = cv2.CascadeClassifier(path2class)

        #Detect multiscale images 
        self.faces = self.haar_face_cascade.detectMultiScale(self.gray_img, scaleFactor=1.5, minNeighbors=6);

    def number_faces(self):
        #print the number of faces found 
        st.text('No of Faces found: ' + str(len(self.faces)))

    def convertToGray(self, img):
        # Convert the RGB  image to grayscale
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    def convertToRGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    
    def detection(self):
    
        faces_crop = []
        for (x, y, w, h) in self.faces:  
            obj = self.img_original[y:y + h, x:x + w]
            faces_crop.append(obj)
            cv2.rectangle(self.img_with_detections, (x, y), (x+w, y+h), (0, 255, 0), 2)
        st.image(self.convertToRGB(self.img_with_detections))  
        return faces_crop
    
        



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # SEE THE FORWARD FUNCTION COMMENTS TO SEE WHERE THE DIMENSIONS OF THE IMAGE COME FROM
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5) # (b,1,96,96) to (b,4,92,92)
        self.conv1_bn = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3) # (b,4,46,46) to (b,64,44,44)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3) # (b,64,22,22) to (b,128,20,20)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3) # (b,128,10,10) to (b,256,8,8)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256*4*4, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 30)
        self.dp1 = nn.Dropout(p=0.4)
       
    def forward(self, x, verbose=False):
        # 1 CONVOLUTIONAL LAYER
        # Input size: 96x96
        # (Hinput_size - Hkernel_size + 1 = Outputsize)
        # Output size = 96-5+1 = 92 
        # Max Pool from 1 Layer
        # Output after Max Pooling window (2,2): (92-2+2)/2 = 46
        x = self.conv1_bn(self.conv1(x))
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dp1(x)
        
        # 2 CONVOLUTIONAL LAYER
        # Input size: 46x46
        # (Hinput_size - Hkernel_size + 1 = Outputsize)
        # Output size = 46-3+1 = 44 
        # Max Pool from 2 Layer
        # Output after Max Pooling window (2,2): (44-2+2)/2 = 22
        x = self.conv2_bn(self.conv2(x))
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dp1(x)
        
        # 3 CONVOLUTIONAL LAYER
        # Input size: 22x22
        # (Hinput_size - Hkernel_size + 1 = Outputsize)
        # Output size = 22-3+1 = 20
        # Max Pool from 3 Layer
        # Output after Max Pooling window (2,2): (20-2+2)/2 = 10
        x = self.conv3_bn(self.conv3(x))
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dp1(x)
        
        # 4 CONVOLUTIONAL LAYER
        # Input size: 10x10
        # (Hinput_size - Hkernel_size + 1 = Outputsize)
        # Output size = 10-3+1 = 8
        # Max Pool from 4 Layer
        # Output after Max Pooling window (2,2): (8-2+2)/2 = 4
        x = self.conv4_bn(self.conv4(x))
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dp1(x)
        
        # END OF THE CONVOLTUTION STAGE
        # 256 outputs of size 4x4
        x = x.view(-1, 256*4*4)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dp1(x)
        x = self.fc3(x)
        return x

keypoints_torch_model = CNN()
keypoints_torch_model.load_state_dict(torch.load(models_dir+'/Facial_KeyPoints_Model', map_location=torch.device('cpu')))



def faces_oneChannel_96(faces_crop, model):
  count=1
  fig = plt.figure(figsize=(10,20))
  for face in faces_crop:
    # must be type = numpy.ndarray
    type(face)
    # We get one channel out of the three channels of the picture
    face = face[:,:,1]
    # we transform it to PIL file so we we can resize it in order to feed it to our model, it only accepts pics of size 96x96
    face = Image.fromarray(face)
    face_96 = face.resize((96,96),Image.ANTIALIAS)

    # then we convert it back to numpy to manipulate it 
    test_face = np.array(face_96)

    # We convert it to torch domain so we can use it in our model
    test_face_torch = torch.from_numpy(test_face).float()
    test_face = test_face_torch.reshape(1,1,96,96) 

    # Using the model to predict the coordinates in the face we are dealing in this iteration
    test_predictions_plantilla = model(test_face)
    test_predictions_plantilla = test_predictions_plantilla.cpu().data.numpy()

    # This is the list with the face keypoints we are detecting
    #keypts_labels_plantilla = train_data.columns.tolist() 

    # We pair the coordinates and pile then in columns for coord x and coord y
    coord = np.vstack(np.split(test_predictions_plantilla[0],15))

    plt.subplot(1,len(faces_crop),count)
    count+=1
    plt.imshow(face_96)
    plt.plot(coord[:,0], coord[:,1], 'o', color='White', label='predicted')
    plt.axis('off')

  st.pyplot(fig)



#### NATURAL LANGUAGE PROCESSING
@st.cache_resource
def load_nlp_model(path_model, path_xgboost_head):
    model_sentence_transformer = SentenceTransformer(path_model)
    classifier = XGBClassifier()
    classifier.load_model(path_xgboost_head)
    return model_sentence_transformer, classifier
model_sentence_transformer, classifier = load_nlp_model(path_model='paraphrase-mpnet-base-v2', path_xgboost_head=models_dir+"/twitter_sentiment_xgb_model.json")








