import pandas as pd
import os
from visualInsights.constants import *
from visualInsights import logger
from visualInsights.utils.utils import read_yaml

config = read_yaml(CONFIG_FILE_PATH)
params = read_yaml(PARAMS_FILE_PATH)

class DataProcessor:

    def __init__(self):
        self.output_path = config.data_output['output_data_path']
        self.nuscenes_version = config.data_loader['nuscenes_version']

    def get_class_distribution(self, data):

        channel_list, iid_list, name_list = [], [], []

        for key, value in data.items():
            channel = value["channel"]
            iID = value["iID"]
            annotations = value["annotations"]

            for annotation in annotations:
                channel_list.append(channel)
                iid_list.append(iID)
                name_list.append(annotation["name"])

        image_df = pd.DataFrame({"channel": channel_list,
                                 "iID": iid_list,
                                 "name": name_list
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

        image_df_filename = os.path.join(self.output_path, self.nuscenes_version+"_image_data.csv")
        class_counts_filename = os.path.join(self.output_path, self.nuscenes_version+"_class_counts.csv")
        sub_categogy_filename = os.path.join(self.output_path, self.nuscenes_version+"_sub_category_counts.csv")

        logger.info("Saving class distribution files.")
        image_df.to_csv(image_df_filename, index=False)
        class_counts.to_csv(class_counts_filename, index=False)
        sub_category_counts.to_csv(sub_categogy_filename, index=False)
