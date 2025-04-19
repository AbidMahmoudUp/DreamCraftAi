import tensorflow as tf
from PIL import Image
import numpy as np
import os
from config.settings import Settings
from utils.logging import setup_logging

logger = setup_logging()

class DiseaseService:
    def __init__(self, settings: Settings):
        self.model = tf.keras.models.load_model("InceptionV3-plantDataset-noAugmentation70.h5",compile=False)
        self.temp_dir = "temp"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        self.class_labels = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]

    def predict_image(self, image_path: str) -> str:
        try:
            image = Image.open(image_path)
            image = image.resize((100, 100))
            input_arr = np.array(image) / 255.0
            input_arr = np.expand_dims(input_arr, axis=0)
            predictions = self.model.predict(input_arr)
            predicted_index = np.argmax(predictions)
            return self.class_labels[predicted_index]
        except Exception as e:
            logger.error(f"Error predicting image: {str(e)}")
            raise

    def process_uploaded_image(self, file_content: bytes, filename: str) -> str:
        file_location = os.path.join(self.temp_dir, filename)
        with open(file_location, "wb") as f:
            f.write(file_content)
        try:
            result = self.predict_image(file_location)
            return result
        finally:
            if os.path.exists(file_location):
                os.remove(file_location)