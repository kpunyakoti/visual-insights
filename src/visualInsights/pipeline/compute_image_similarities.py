import os
import sys
import numpy as np
import tqdm
import shutil
from visualInsights.constants import *
from visualInsights import logger
from visualInsights.utils.utils import *
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3 as db

config = read_yaml(CONFIG_FILE_PATH)
params = read_yaml(PARAMS_FILE_PATH)

class computeImageSimilarity:

    def __init__(self):

        self.image_dir = os.path.join(config.data_loader['nuscenes_data_path'],config.data_loader['nuscenes_version'],"samples/CAM_FRONT")
        self.similar_img_dir = os.path.join(config['data_output']['output_data_path'],"similar_images")
        self.db_file_path = os.path.join(config.db_config['db_path'], config.db_config['db_name']+".db")
        self.query_image = params.query_image_name
        self.n = params.n_similar_images

        self.conn = db.connect(self.db_file_path)
        self.cursor = self.conn.cursor()
        logger.info("Connected to feature vector database - "+config.db_config['db_name']+".db")

    def retrieve_feature_vector(self, img_name):
        # Get the row
        self.cursor.execute("SELECT * FROM ImageFeatures WHERE img_name = ?", (img_name,))
        result = self.cursor.fetchone()

        if result is not None:
            # Convert the bytes back to a float array and return it
            float_array_bytes = result[1]
            float_array = np.frombuffer(float_array_bytes, dtype='float32')
            return float_array
        else:
            return None

    def compute_top_n_similar_images(self):

        self.cursor.execute('SELECT img_name FROM ImageFeatures')
        img_names = self.cursor.fetchall()

        self.cursor.execute("DROP TABLE IF EXISTS SimilarityResults")

        self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS SimilarityResults (
                img_name TEXT,
                similar_images TEXT
                )
                ''')

        for img_name in tqdm.tqdm(img_names):
            # Calculate similarity scores for img_name with all other images
            similarity_scores = []

            for other_img_name in img_names:
                if img_name != other_img_name:  # Exclude the image itself
                    similarity_score = cosine_similarity([self.retrieve_feature_vector(other_img_name[0])],
                                                         [self.retrieve_feature_vector(img_name[0])]).flatten()[0]
                    similarity_scores.append((other_img_name[0], similarity_score))

            # Sort the images by similarity score in descending order
            sorted_images = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

            # Select the top N similar images (excluding the image itself)
            top_n_similar_images = [img[0] for img in sorted_images[:self.n]]

            self.cursor.execute('INSERT INTO SimilarityResults (img_name, similar_images) VALUES (?, ?)',
                           (img_name[0], ','.join(top_n_similar_images)))

        self.conn.commit()

        logger.info("Generated top n similar images for each image and stored in SimilarityResults table.")

    def empty_directory(self, dir_path):
        try:
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)

                # Check if the item is a file and remove it
                if os.path.isfile(item_path):
                    os.remove(item_path)

                # Check if the item is a directory and remove it recursively
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        except FileNotFoundError:
            sys.exit(f"Please check if the directory '{self.similar_img_dir}' exists. Exiting!!")

    def get_similar_images(self):

        #empty similar image output directory before copying new similar images.
        self.empty_directory(self.similar_img_dir)

        self.cursor.execute("select similar_images from SimilarityResults where img_name =?", (self.query_image,))
        result = self.cursor.fetchall()

        if result is not None:
            for similar_image in result[0][0].split(","):
                img_path = os.path.join(self.image_dir, similar_image)
                shutil.copy2(img_path, self.similar_img_dir)
            logger.info(f"Retrieved top {self.n} similar images for the given image!")
        else:
            logger.info("No similar images found for the given image!")
