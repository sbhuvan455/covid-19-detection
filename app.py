import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import keras
import keras.utils as image

def predict_image(image_path, model):
    xtest_image = image.load_img(image_path, target_size=(224, 224))
    xtest_image = image.img_to_array(xtest_image)
    xtest_image = np.expand_dims(xtest_image, axis=0)
    results = model.predict(xtest_image)

    imggg = cv2.imread(image_path)
    imggg = np.array(imggg)
    imggg = cv2.resize(imggg, (400, 400))

    if results[0][0] == 0:
        prediction = 'Positive For Covid-19'
    else:
        prediction = 'Negative for Covid-19'

    return imggg, prediction

def main():
    st.title("Covid-19 X-ray Image Classifier")

    uploaded_file = st.file_uploader("Choose an X-ray image")

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        with open("temp_image.jpg", "wb") as f:
            f.write(image_data)

        model = tf.keras.models.load_model("model.h5")

        imggg, prediction = predict_image("temp_image.jpg", model)

        st.image(imggg)
        st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()
