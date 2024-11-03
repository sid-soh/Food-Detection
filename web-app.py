import argparse
import io
from PIL import Image
import datetime

import torch
import cv2 
import numpy as np
import tensorflow as tf
from re import DEBUG, sub 
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os 
import subprocess 
from subprocess import Popen
import re 
import requests
import shutil 
import time
import glob 

from ultralytics import YOLO

app = Flask(__name__)

@app.route("/")
# Load the website
def start_web():
    return render_template("main.html")

@app.route("/", methods=["GET", "POST"])
def food_recognition():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            # Storing data in the uploads folder
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            print("Upload folder: " + filepath)
            f.save(filepath)
            global imgpath

            food_recognition.imgpath = f.filename
            print("Printing food_reocgnition :::::" + food_recognition)

            file_extension = f.filename.rsplit('.', 1)[1].lower()
            img = cv2.imread(filepath)

            if file_extension == 'jpg' or file_extension == 'png':
                frame = cv2.imencode('.'+file_extension, cv2.UMat(img))[1].tobytes()
            else: # invalid file format
                return 0

            image = Image.open(io.BytesIO(frame))
            yolo = YOLO("insert.yolo.file.here")
            detections = yolo.predict(image, save=True)
            return display(f.filename) # displays the yolo model boxes
        
@app.route('/<path:filename>')
def display(filename):
    # retrieve the correct file to display
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = folder_path + "/" + latest_subfolder
    print("Printing directory: ", directory)
    files = os.listdir(directory)
    latest_file = files[0]

    print(latest_file)

    filename = os.path.join(folder_path, latest_subfolder, latest_file)

    environ = request.environ

    return send_from_directory(directory, latest_file, environ) # result is shown in a separate tab
