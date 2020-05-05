#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import imutils
import shutil
import torch
import cv2
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import torchvision.models as models
import torch.utils.model_zoo
import seaborn as sns
import pickle
import tkinter
from tkinter import filedialog, messagebox

def show_error(text):
    tkinter.messagebox.showerror('Error', text)

def show_results(text):
    tkinter.messagebox.showinfo("Results", text)

def MRI_Process(file):
    file = file[0]
    # If the file ends with ".jpeg", process it.
    if os.path.abspath(file).endswith('.jpeg') or os.path.abspath(file).endswith('.jpg'):
            VGG_model = pickle.load(open("VGG.sav", 'rb'))
            #NN_model = pickle.load(open("NN.sav", 'rb'))
            #CNN_model = pickle.load(open("CNN.sav", 'rb'))

            img = load_edge_crop(file)

            img = transforms.Compose([transforms.Grayscale(num_output_channels=3),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])])(Image.fromarray(img))

            img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])

            tumor = VGG_model(img)
            pred = tumor.max(1, keepdim=True)[1]

            if pred == 1:
                show_results("Tumor")
            else:
                show_results("No Tumor")

    # If the file ends with something other than ".jpeg", show error.
    else:
        show_error("The file %s is not a .jpeg or .jpg file and will be ignored." % (file))

def load_edge_crop(imagename):

        img = cv2.imread(imagename,0)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(img, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        new_img = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]].copy()
        new_img = cv2.resize(new_img,dsize=(32,32))
        return new_img

# Create the Tkinter interface.
app = tkinter.Tk()
app.geometry('275x175')
app.title("MRI Prediction")

# Open the file picker and send the selected Image.
def clicked():
    t = tkinter.filedialog.askopenfilenames()
    MRI_Process(t)

# Create the window header.
header = tkinter.Label(app, text="MRI Prediction(Select Image)", fg="blue",
    font=("Arial Bold", 16))
header.pack(side="top", ipady=10)

# Add the descriptive text.
text = tkinter.Label(app,
    text="Select Brain Image to determine if person have tumor or not")
text.pack()

# Draw the button that opens the file picker.
open_files = tkinter.Button(app, text="Select Image...", command=clicked)
open_files.pack(fill="x")

# Initialize Tk window.
app.mainloop()
