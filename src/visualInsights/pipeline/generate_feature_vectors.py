import os
import tqdm
import numpy as np
from visualInsights.constants import *
from visualInsights import logger
from visualInsights.utils.utils import *
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
import sqlite3 as db

config = read_yaml(CONFIG_FILE_PATH)

class generateFeatureVectorDB:

    def __init__(self):

        self.image_dir = os.path.join(config.data_loader['nuscenes_data_path'],config.data_loader['nuscenes_version'],"samples/CAM_FRONT")
        self.db_file_path = os.path.join(config.db_config['db_path'], config.db_config['db_name']+".db")

        self.base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('fc2').output)

        self.conn = db.connect(self.db_file_path)
        self.cursor = self.conn.cursor()
        logger.info("Created & connected to feature vector database - "+config.db_config['db_name']+".db")

    def get_image_array(self, img_path):

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = np.expand_dims(image.img_to_array(img), axis=0)
        return preprocess_input(img_array)

    def compute_feature_vector(self, img_path):

        img_array = self.get_image_array(img_path)
        feature_vector = self.model.predict(img_array, verbose=0)
        return feature_vector.flatten()

    def store_feature_vector(self, img_name, feature_vector):
        # Convert the feature vector as bytes and store it into the database
        fv_bytes = feature_vector.tobytes()
        self.cursor.execute("INSERT INTO ImageFeatures (img_name, fV)  VALUES (?, ?)", (img_name, fv_bytes))
        self.conn.commit()

    def check_in_db(self, img_name):
        # Retrieve the record from the database
        self.cursor.execute("SELECT * FROM ImageFeatures WHERE img_name = ?", (img_name,))
        result = self.cursor.fetchone()
        return False if result is None else True

    def generate_feature_vectors(self):

        #create ImageFeatures table
        self.cursor.execute("CREATE TABLE IF NOT EXISTS ImageFeatures (img_name TEXT, fV BLOB)")

        for img in tqdm.tqdm(os.listdir(self.image_dir)):
            img_path = os.path.join(self.image_dir, img)

            if not self.check_in_db(img):
                feature_vector = self.compute_feature_vector(img_path)
                self.store_feature_vector(img, feature_vector)

        logger.info("Generated feature vectors and stored in ImageFeatures table.")
