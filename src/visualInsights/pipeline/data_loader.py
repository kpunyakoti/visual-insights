import os
import tqdm
from pathlib import  Path
from visualInsights.constants import *
from visualInsights import logger
from visualInsights.utils.utils import *
from nuscenes.nuscenes import NuScenes

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
        nuscenes_data_path = os.path.join(self.data_path,self.nuscenes_version)
        nuscenes_obj = NuScenes(version=self.nuscenes_version, dataroot=nuscenes_data_path, verbose=True)
        return nuscenes_obj

class nuscenesDataExtractor:
    def __init__(self, nuscenes_obj):
        
        self.nuscenes_obj = nuscenes_obj
        self.nuscenes_version = config.data_loader['nuscenes_version']
        self.output_path = config.data_output['output_data_path']
        self.channel_names = params.channel_names

    def extract_nuscenes(self):
        logger.info(f"Finished Loading Dataset")
        logger.info("Analysing Annotations...")

        image_data_dict = dict()

        for scene in tqdm.tqdm(self.nuscenes_obj.scene):
            token = scene['first_sample_token']
            while token != '':
                sample = self.nuscenes_obj.get('sample', token)
                annotation_list = sample['anns']

                for channel in self.channel_names:
                    channel_token = sample['data'][channel]
                    # Get the annotations for this sample
                    file_path, boxes, camera_intrinsic = self.nuscenes_obj.get_sample_data(channel_token,
                                                                             selected_anntokens=annotation_list)

                    # If this same was not there in the list already add an empty array to hold the annotations
                    if channel_token not in image_data_dict:
                        image_data_dict[channel_token] = {'channel': channel,
                                                          'iID': channel_token,
                                                          'filename': os.path.basename(file_path),
                                                          'annotations': []}

                    # For each of the annotations in that sample
                    for box in boxes:
                        image_data_dict[channel_token]['annotations'].append({'name': box.name
                                                                              })
                # Get the next token
                token = sample['next']

        logger.info("Saving extracted image data dict")
        json_filename = os.path.join(self.output_path, self.nuscenes_version+"_camera_boxes.json")
        save_json(Path(json_filename), image_data_dict)
