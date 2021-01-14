import cv2
from pnslib import utils
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image


def face_reg(img):
    # load face cascade
    face_cascade = cv2.CascadeClassifier(
        utils.get_haarcascade_path('haarcascade_frontalface_default.xml'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (225, 0, 0), 4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    #img =  cv2.resize(img, (1024,768))
    return faces, img


st.title("Detect faces")
st.markdown('Please use .jpg , .jpeg and .png file')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", 'png'])

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Before", use_column_width=True)
    st.write("")
    st.write("Detecting faces...")
    box, image_boxed = face_reg(image)
    if len(box) == 0:
        st.write("No faces detected")
    else:
        st.image(image_boxed, caption="After", use_column_width=True)