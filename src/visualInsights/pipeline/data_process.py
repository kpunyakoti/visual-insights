"""Process the nuscenes data to generate class distribution and performance scores."""
import os
import sys
import pandas as pd
from pathlib import Path
from visualInsights.constants import *
from visualInsights import logger
from visualInsights.utils.utils import read_yaml, load_json

config = read_yaml(CONFIG_FILE_PATH)
params = read_yaml(PARAMS_FILE_PATH)

class DataProcessor:

    def __init__(self):

        #input files
        self.data_path = config.data_loader['nuscenes_data_path']
        self.nuscenes_version = config.data_loader['nuscenes_version']
        self.leaderboard_file = os.path.join(self.data_path, "leaderboard.json")

        #output_files
        self.output_path = config.data_output['output_data_path']
        self.class_counts_file = os.path.join(self.output_path, self.nuscenes_version + "_class_counts.csv")
        self.image_df_file = os.path.join(self.output_path, self.nuscenes_version + "_image_data.csv")
        self.metrics_file = os.path.join(self.output_path, self.nuscenes_version + "_final_metrics.csv")

    def get_class_distribution(self):
        try:
            json_filename = os.path.join(self.output_path, self.nuscenes_version + "_camera_boxes.json")
            data = load_json(Path(json_filename))
        except FileNotFoundError:
            sys.exit(f"{json_filename} does not exist! Please check the paths and file names. Exiting!!")

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
        image_df['name'] = image_df['name'].str.replace(r'(?<=[^.])$', '.')
        image_df[['class_name', 'sub_category', 'grain']] = image_df['name'].str.split('.', expand=True, n=2)

        class_counts = pd.DataFrame({'counts': image_df[['class_name', 'sub_category']].value_counts(),
                                            'proportion': image_df[['class_name', 'sub_category']].value_counts(
                                                normalize=True)
                                            }).reset_index()

        class_counts = class_counts[['class_name', 'sub_category', 'counts', 'proportion']]
        class_counts.loc[class_counts['sub_category']=="", 'sub_category'] = class_counts['class_name']
        
        logger.info("Saving class distribution files.")
        image_df.to_csv(self.image_df_file, index=False)
        class_counts.to_csv(self.class_counts_file, index=False)

    def get_performance_scores(self):

        leaderboard = load_json(Path(self.leaderboard_file))
        class_counts_df = pd.read_csv(self.class_counts_file)

        class_counts_df.loc[class_counts_df['sub_category'] == 'construction', 'sub_category'] = 'construction_vehicle'
        class_counts_df.loc[class_counts_df['sub_category'] == 'trafficcone', 'sub_category'] = 'traffic_cone'

        mean_scores = {}
        for i in range(5):
            # Calculate mean performance score of top 5 submissions
            metric_summary = leaderboard[i]['metrics_summary']['mean_dist_aps']
            for key, value in metric_summary.items():
                if key in mean_scores:
                    mean_scores[key].append(value)
                else:
                    mean_scores[key] = [value]

        # Calculate the mean for each object
        for key, value in mean_scores.items():
            mean_scores[key] = sum(value) / len(value)

        # Create a DataFrame from the mean scores
        leader_df = pd.DataFrame(mean_scores.items(), columns=['sub_category', 'mean_aps_score'])

        # Merge class counts and performance scores
        final_metrics_df = class_counts_df.merge(leader_df, on = 'sub_category', how='inner')

        final_metrics_df.rename(columns={'class_name': 'class_group', 'sub_category': 'class_name'}, inplace=True)
        final_metrics_df.to_csv(self.metrics_file, index=False)
