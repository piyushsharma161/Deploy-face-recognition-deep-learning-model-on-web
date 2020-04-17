#!/usr/bin/env python
# coding: utf-8



from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
#from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
#from keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('C:/projects/face_reco/Deep-Learning-Face-Recognition/haarcascade_frontalface_default.xml')

model = load_model('C:/projects/face_reco/facefeatures_new_model.h5')
model._make_predict_function() 



def face_extractor(img):
    faces = face_classifier.detectMultiScale(img, 1.3, 5) 
    faces_cor = []
    face_imgs = np.empty((len(faces), 224, 224, 3))
    
    for i, face in enumerate(faces):  
        if face is ():
            return None, None
        # Crop all faces found
        for (x,y,w,h) in face.reshape((1,4)):
            face_cor = []
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
            x=x-10
            y=y-10
            face_cor.append(x)
            face_cor.append(y)
            face_img = img[y:y+h+50, x:x+w+50]
            try:
                face_img = cv2.resize(face_img, (224, 224))           
                face_imgs[i,:,:,:] = face_img
                faces_cor.append(face_cor)
            except Exception as e:
                return None, None

    return faces_cor, face_imgs



def model_predict(img_path):
    result = 'Not detected'
    img = cv2.imread(img_path)
    faces_cor, faces = face_extractor(img)
    img = img/255.0
    if faces is not None:
        if faces.shape[0]>0:
            for i in range(len(faces)):
                face = faces[i]
                face_cor = faces_cor[i]
                face = face/255.0
                face = face.reshape([1,224,224,3])
                pred = model.predict(face)
                #print(pred)
                #name="None matching"
                if(pred[0][1]>0.5):
                    return 'Piyush'
                    #cv2.putText(img,name, (face_cor[0], face_cor[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                elif (pred[0][0]>0.5):
                    return 'Myra'       
                    #cv2.putText(img,name, (face_cor[0], face_cor[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        else:
            return 'Not detected'
            #cv2.putText(img,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

    else:
        return 'Not detected'
        #cv2.putText(img,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2) 





@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path)

        return result
    return None


if __name__ == '__main__':
    app.run(debug=False, threaded=False)

