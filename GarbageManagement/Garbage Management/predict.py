from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image 
import numpy as np

model = load_model("garbage_classifier_model.h5")
class_names = ['Battery', 'Biological', 'Cardboard', 'Clothes', 'Glass', 'Metal', 'Paper', 'Plastic', 'Shoes', 'Trash']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    return class_names[np.argmax(pred)], np.max(pred)