import os
import sys
import numpy as np
import tqdm
import pandas as pd
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

        #fetch feature vectors for all images
        self.cursor.execute('SELECT img_name FROM ImageFeatures')
        img_names = self.cursor.fetchall()

        img_feature_dict = {img[0]: np.array(self.retrieve_feature_vector(img[0]).tolist()) for img in img_names}
        image_feature_df = pd.DataFrame(list(img_feature_dict.items()), columns=['image_name', 'feature_vector'])

        self.cursor.execute("DROP TABLE IF EXISTS SimilarityResults")
        self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS SimilarityResults (
                img_name TEXT,
                similar_images TEXT
                )
                ''')

        #compute similarity matrix for all images
        cosine_sim = cosine_similarity(image_feature_df['feature_vector'].tolist(),
                                       image_feature_df['feature_vector'].tolist())

        result_df = pd.DataFrame(columns=['image_name', 'top_similar_images'])

        # Iterate through each image
        for i, image_name in tqdm.tqdm(enumerate(image_feature_df['image_name'])):
            # Get the similarity scores for the current image
            similar_scores = cosine_sim[i]
            # Sort in descending order and get the indices of the top n similar images (excluding the image itself)
            top_indices = (-similar_scores).argsort()[1:self.n+1]
            # Get the image names corresponding to the top indices
            top_similar_images = image_feature_df['image_name'].iloc[top_indices].tolist()
            # Append to the result DataFrame
            result_df = result_df.append({'image_name': image_name, 'top_similar_images': top_similar_images},
                                         ignore_index=True)

        result_df['top_similar_images'] = result_df['top_similar_images'].apply(lambda x: ','.join(x))

        for index, row in result_df.iterrows():
            self.cursor.execute("INSERT OR REPLACE INTO SimilarityResults (img_name, similar_images) VALUES (?, ?)",
                           (row['image_name'], row['top_similar_images']))

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
