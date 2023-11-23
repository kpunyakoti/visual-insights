import os
import tqdm
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

class nuscenesDataExtractor:
    def __init__(self, nuscene_obj):
        self.nuscene_obj = nuscene_obj
        self.nuscenes_version = config.data_loader['nuscenes_version']
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
                        #corners = view_points(box.corners(), np.array(camera_intrinsic), normalize=True)[:2, :]
                        image_data_dict[channel_token]['annotations'].append({'name': box.name#,
                                                                              #'corners': corners.tolist()
                                                                              })
                # Get the next token
                token = sample['next']

        logger.info("Saving extracted image data dict")
        json_filename = os.path.join(self.output_path, self.nuscenes_version+"_camera_boxes.json")
        save_json(Path(json_filename), image_data_dict)
