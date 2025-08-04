import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20

#Data Preparation
#Normalization: Rescaling beacause of fast processing 

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

#training dataset
train_data = datagen.flow_from_directory(
    'garbage-dataset',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

#validation dataset
val_data = datagen.flow_from_directory(
    'garbage-dataset',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size= BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

#CNN Model
#Convolution: Layer to extract more features
#MaxPooling: Layer which extracts more  and important features
#Flatten: Arranging the features in 1D Array
#Full Connection: Adding Dense Layers i.e., Hidden Layers 
#We are using list DS to store all the layers inn an order
#Conv2D(32, (3,3), activation='relu', input_shape =(IMG_SIZE, IMG_SIZE, 3)): 32 is batch_size, (3,3) is feature map size, 
#relu:activation fn, Input shape(128,128, 3(For Color Image- RGB))

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape =(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

#model compiling, and calcualting the loss and accuracy of the model then fitting the model with training data and val_data as testing data, 10 epochs
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data= val_data, epochs=EPOCHS)

#Saving the model, For Deep Learning save model as h5
model.save("garbage_classifier_model.h5")