import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# Some utilites
import numpy as nps
from util import base64_to_pil
from pneumonia_detection import disease

# Declare a flask app
app = Flask(__name__)


# Model saved with Keras model.save()
MODEL_PATH = 'models/dense_net_pretrained.h5'

model=disease(MODEL_PATH)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        img.save("./uploads/image.png")

        # Make prediction
        preds = model.predict('./uploads/image.png')
        return jsonify(result=f"Probability of Pneumonia {preds*100:.3f}%")


if __name__ == '__main__':

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
