from flask import Flask, render_template, redirect, request, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import os
import tensorflow
import numpy as np
from keras.utils import load_img, img_to_array
import keras
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'bucket')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}

MODEL = tensorflow.keras.models.load_model(os.path.join(MODEL_DIR, 'model.h5'))

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CLASSES = ['Apple scab',
           'Apple Black rot',
           'Apple Cedar apple rust',
           'Apple healthy', 
           'Blueberry healthy',
           'Cherry (including sour) Powdery mildew', 
           'Cherry (including sour) healthy', 
           'Corn (maize) Cercospora leaf spot Gray leaf spot', 
           'Corn(maize) Common rust',
           'Corn(maize) Northern Leaf Blight', 
           'Corn(maize) healthy', 
           'Grape Black rot', 
           'Grape Esca(Black Measles)', 
           'Grape Leaf blight(Isariopsis Leaf Spot)', 
           'Grape healthy',
           'Orange Haunglongbing(Citrus greening)', 
           'Peach Bacterial spot', 
           'Peach healthy', 
           'Pepper Bell Bacterial_spot', 
           'Pepper Bell healthy', 
           'Potato Early blight', 
           'Potato Late blight', 
           'Potato healthy', 
           'Raspberry healthy', 
           'Soybean healthy', 
           'Squash Powdery mildew', 
           'Strawberry Leaf scorch', 
           'Strawberry healthy', 
           'Tomato Bacterial spot', 
           'Tomato Early blight', 
           'Tomato Late blight', 
           'Tomato Leaf Mold', 
           'Tomato Septoria leaf spot', 
           'Tomato Spider mites (Two-spotted spider mite)', 
           'Tomato Target Spot', 
           'Tomato Yellow Leaf Curl Virus', 
           'Tomato mosaic virus', 'Tomato healthy']

disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

@app.route('/')
def home():
        return "Welcome To Api"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/plantdisease', methods=['GET', 'POST'])
def plantdisease():
    disease_name = ""
    description = ""
    prevent = ""
    image_url = ""
    supplement_name = ""
    supplement_image_url = ""
    supplement_buy_link = ""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            model = MODEL
            # imagefile = keras.utils.load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(224, 224, 3))
            # input_arr = keras.preprocessing.image.img_to_array(imagefile)
            imagefile = load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(224, 224, 3))
            input_arr = img_to_array(imagefile)
            input_arr = np.array([input_arr])
            result = model.predict(input_arr)
            probability_model = tensorflow.keras.Sequential([model, 
                                         tensorflow.keras.layers.Softmax()])
            predict = probability_model.predict(input_arr)
            pred = np.argmax(predict[0])
            res = CLASSES[pred]
            disease_name = res
            description =disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]

    return {
         "disease_name":disease_name,
         "description":description,
         "prevent":prevent,
         "image_url":image_url,
         "supplement_name":supplement_name,
         "supplement_image_url":supplement_image_url,
         "supplement_buy_link":supplement_buy_link
    }

# if __name__== "__main__":
#     app.run(debug=True)
