from flask import Flask,request,jsonify,render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

model = load_model("C:\Users\sanjith shalu\OneDrive\Desktop\emotions-based\notebook\ResNet50V2.h5")
Emotions_Classes = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
music_player = pd.read_csv('C:\\Users\\sanjith shalu\\OneDrive\\Desktop\\emotions-based\\datasets\\data_moods.csv')

def load_and_prep_image(filename,image_shape= 224):


    img = cv2.imread(filename)
    GrayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CaseCodeClassifier('./datasets/haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(GrayImg,1.1,4)

    for x,y,w,h in faces:
        roi_GrayImg = GrayImg[y:y+h,x:x+w]
        roi_img =img[y:y+h,x:x+w]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        plt.imshow(cv2.cvtColor(img.cv2.COLOR_BGR2RGB))

        faces = faceCascade.detectMultiScale(roi_img,1.1,4)

        if len(faces)==0:
            print("no Faces Detected")
        else:
            for(ex,ey,ew,eh) in faces:
                img = roi_img[ey:ey+eh,ex:ew+ew]
    
    RGBImg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    RGBImg = cv2.resize(RGBImg,(image_shape,image_shape))

    RGBImg = RGBImg/255.

    return RGBImg

def pred_and_recommend(filename):
    img = load_and_prep_image(filename)

    pred = model.predict(np.extend_dims(img,axis = 0))

    pred_class = Emotions_Classes[np.argmax(pred)]

    Recommod = Recommod_Songs(pred_class)

    return pred_class,Recommod



def Recommod_Songs(pred_class):
    Recommod=[]
    if(pred_class == 'Disgust'):
        Recommod = get_music_recommendations('sad')
    elif pred_class in ['Happy','Sad']:
        Recommod = get_music_recommendations('Happy')
    elif pred_class in ['Fear','Angry']:
        Recommod = get_music_recommendations('Calm')
    elif pred_class in ['surprise','Neutral']:
        Recommod = get_music_recommendations('Energetic')

def get_music_recommendations(mood):
    play = music_player[music_player['mood'] == mood]
    play = play.sort_values(by = 'popularity',ascending=False)[:5]
    Recommod = play[['album','artist','name','popularity','release_date']].to_dict(orient = 'recommod')
    return Recommod

@app.route('/')
def home():
    return render_template('index.html') 
@app.route('/predict',methods = ['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No File Found'}),400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No File Selected'}),400

    filename = secure_filename(file.filename)
    file_path = os.path.join('./uploads',filename)
    file.save(file_path)

    Emotions,Recommod = pred_and_recommend(file_path)

if __name__ == '__main__':
    app.run(debug=True,port = 8080)