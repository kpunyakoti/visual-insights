import isoduration.parser.parsing
import numpy as np
import pandas as pd
import tqdm
import os
from pathlib import  Path
from visualInsights.constants import *
from visualInsights import logger
from visualInsights.utils.utils import read_yaml, save_json
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points

config = read_yaml(CONFIG_FILE_PATH)
params = read_yaml(PARAMS_FILE_PATH)

# Define Data Loader
class DataLoader:
    def __init__(self, source = "nuscenes"):

        if source == "nuscenes":
            self.data_path = config.data_loader['nuscenes_data_path']
            self.nuscenes_version = config.data_loader['nuscenes_version']
        else:
            self.data_path = config.data_loader['cifar_data_path']

    def load_nuscenes_data(self):

        nuscene_obj = NuScenes(version=self.nuscenes_version, dataroot=self.data_path, verbose=True)
        return nuscene_obj

class nuscenesDataProcesser:
    def __init__(self, nuscene_obj):
        self.nuscene_obj = nuscene_obj
        self.output_path = config.data_output['output_data_path']
        self.channel_names = params.channel_names

    def extract_nuscenes(self):
        logger.info(f"Finished Loading Dataset")
        logger.info("Analysing Annotations...")

        image_data_dict = dict()

        for scene in tqdm.tqdm(self.nuscene_obj.scene):
            token = scene['first_sample_token']
            while token != '':
                sample = self.nuscene_obj.get('sample', token)
                annotation_list = sample['anns']

                for channel in self.channel_names:
                    channel_token = sample['data'][channel]
                    # Get the annotations for this sample
                    filename, boxes, camera_intrinsic = self.nuscene_obj.get_sample_data(channel_token,
                                                                             selected_anntokens=annotation_list)

                    # If this same was not there in the list already add an empty array to hold the annotations
                    if channel_token not in image_data_dict:
                        image_data_dict[channel_token] = {'channel': channel,
                                                          'iID': channel_token,
                                                          'filePath': filename,
                                                          'annotations': []}

                    # For each of the annotations in that sample
                    for box in boxes:
                        corners = view_points(box.corners(), np.array(camera_intrinsic), normalize=True)[:2, :]
                        image_data_dict[channel_token]['annotations'].append({'name': box.name,
                                                                              'corners': corners.tolist()
                                                                              })
                # Get the next token
                token = sample['next']

        logger.info("Saving extracted image data dict")
        json_file_path = os.path.join(self.output_path, "camera_boxes.json")
        save_json(Path(json_file_path), image_data_dict)

        return image_data_dict

    def get_class_distribution(self, image_dict):

        data = image_dict.copy()
        channel_list, iid_list, name_list, corners_list = [], [], [], []

        for key, value in data.items():
            channel = value["channel"]
            iID = value["iID"]
            annotations = value["annotations"]

            for annotation in annotations:
                name = annotation["name"]
                corners = annotation["corners"]

                # Append the data to the respective lists
                channel_list.append(channel)
                iid_list.append(iID)
                name_list.append(name)
                corners_list.append(corners)

        image_df = pd.DataFrame({"channel": channel_list,
                                 "iID": iid_list,
                                 "name": name_list,
                                 "corners": corners_list
                                 })
        image_df[['class_name', 'sub_category', 'grain']] = image_df['name'].str.split('.', expand=True, n=2)

        class_counts = pd.DataFrame({'counts': image_df.class_name.value_counts(),
                                     'proportion': image_df.class_name.value_counts(normalize=True)
                                     }).reset_index().rename(columns={"index": "class_name"})

        class_counts = class_counts[['class_name', 'counts', 'proportion']]

        sub_category_counts = pd.DataFrame({'counts': image_df[['class_name', 'sub_category']].value_counts(),
                                            'proportion': image_df[['class_name', 'sub_category']].value_counts(
                                                normalize=True)
                                            }).reset_index()

        sub_category_counts = sub_category_counts[['class_name', 'sub_category', 'counts', 'proportion']]

        image_df_filename = os.path.join(self.output_path, "image_data.csv")
        class_counts_filename = os.path.join(self.output_path, "class_counts.csv")
        sub_categogy_filename = os.path.join(self.output_path, "sub_category_counts.csv")

        logger.info("Saving class distribution files.")
        image_df.to_csv(image_df_filename, index=False)
        class_counts.to_csv(class_counts_filename, index=False)
        sub_category_counts.to_csv(sub_categogy_filename, index=False)
