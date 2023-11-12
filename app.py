
from copy import copy
import io
import os
import sys
import time

import streamlit as st
import cv2 
import keras
import numpy as np
import urllib.request

from keras.layers import (Activation, Conv2D, Dense, Flatten,
                          GlobalAveragePooling2D, Input, MaxPooling2D, Dropout)
from keras.models import Model, Sequential
from keras.optimizers import Adam
# from tensorflow.keras.optimizers.legacy import Adam

from PIL import Image, ImageFile
# from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

train_dir = 'Training'
val_dir = 'Testing'

batch_size = 64
num_epoch = 50

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

def build_model(input_shape):
    model = Sequential()
    # Convolutional layer with 32 filters, each 3x3 in size
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional layer with 64 filters, each 3x3 in size
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional layer with 128 filters, each 3x3 in size
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional layer with 264 filters, each 3x3 in size
    model.add(Conv2D(264, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Try to experiment with the following two lines uncomment one and comment out the other to see the differences in performance
    # model.add(Flatten())  # Flatten the 3D output to 1D
    model.add(GlobalAveragePooling2D())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer with 1 neuron (since it's binary classification with Happy-1 and Sad-0)
    # model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(6, activation='softmax'))

    return model

# def build_model(input_shape):
    # # input_layer = Input(shape=input_shape)
    # # x = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1))(input_layer)
    # # x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    # # x = MaxPooling2D(pool_size=(2, 2))(x)
    # # x = Dropout(0.25)(x)
    
    # # x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    # # x = MaxPooling2D(pool_size=(2, 2))(x)
    # # x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    # # x = MaxPooling2D(pool_size=(2, 2))(x)
    # # x = Dropout(0.25)(x)
    
    # # x = Flatten()(x)
    # # x = Dense(1024, activation='relu')(x)
    # # x = Dropout(0.5)(x)
    # # x = Dense(6, activation='softmax')(x)
    
    # input_layer = Input(shape=input_shape)
    # # x = Conv2D(16, (3, 3), padding='same')(input_layer)
    # # x = Activation('relu')(x)
    # # x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # # x = Conv2D(32, (3, 3), padding='same')(input_layer)
    # # x = Activation('relu')(x)
    # # x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # # x = Conv2D(64, (3, 3), padding='same')(x)
    # # x = Activation('relu')(x)
    # # x = MaxPooling2D(pool_size=(2, 2))(x)

    # # x = Conv2D(128, (3, 3), padding='same')(x)
    # # x = Activation('relu')(x)
    # # x = MaxPooling2D(pool_size=(2, 2))(x)

    # # x = GlobalAveragePooling2D()(x)

    # # x = Flatten()(x)
    # # x = Dense(512)(x)
    # # x = Activation('relu')(x)
    # # x = Dropout(0.5)(x)

    # # # # x = Dense(1, activation='sigmoid')(x)  # Binary classification
    # # # # x = Dense(2, activation='softmax')(x)
    # # x = Dense(6, activation='softmax')(x)


    # model = Model(inputs=input_layer, outputs=x)
    # return model

input_shape = (48,48, 1)
model = build_model(input_shape)
# # model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# # model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
# model_info = model.fit(
#         train_generator,
#         batch_size=batch_size,
#         epochs=num_epoch,
#         # validation_data=validation_generator,
#         callbacks=[reduce_lr])
# model.load_weights('model1.h5')

# model.evaluate(validation_generator, batch_size=batch_size)

# model.save_weights('model1.h5')

model.load_weights('model1.h5')

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
emotion_dict = {0: "Angry", 1: "Fear", 2: "Happy", 3: "Neutral", 4: "Sad", 5: "Surprised"}


st.header("Emotion Recognition System")

img = None

option = st.selectbox(
   "Choose a method for uploading images:",
   ("From URL", "From Computer"),
   index=None,
   placeholder="Select a method...",
)

if(option=="From URL"):
    img_URL = st.text_input('Enter a URL:')
    
    if img_URL:
        req = urllib.request.Request(
            img_URL, 
            data=None, 
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
        })

        url_response = urllib.request.urlopen(req)
        o_img = Image.open(url_response)
        o_img = np.array(o_img)
        img = copy(o_img)
    
if(option=="From Computer"):
    uploaded_img = st.file_uploader('Choose an image', type=['jpg', 'png'])
    if uploaded_img:
        o_img = Image.open(uploaded_img)
        o_img = np.array(o_img)
        img = copy(o_img)

    
if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(o_img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(o_img, emotion_dict[maxindex], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    st.image(o_img)


# Using cv2.imshow() method 
# Displaying the image 
# cv2.imshow("predicted result", img) 

# # waits for user to press any key 
# # (this is necessary to avoid Python kernel form crashing) 
# cv2.waitKey(0) 
  
# # closing all open windows 
# cv2.destroyAllWindows() 

# cv2.imwrite("output.png", img) 

