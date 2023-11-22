import numpy as np
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
        self.camera_name = params.camera_name

    def extract_nuscenes(self):
        logger.info(f"Finished Loading Dataset")
        logger.info("Analysing Annotations...")

        image_data_dict = dict()
        for scene in tqdm.tqdm(self.nuscene_obj.scene):
            token = scene['first_sample_token']
            while token != '':
                sample = self.nuscene_obj.get('sample', token)
                camera_token = sample['data'][self.camera_name]
                annotation_list = sample['anns']

                # Get the annotations for this sample
                filename, boxes, camera_intrinsic = self.nuscene_obj.get_sample_data(camera_token,
                                                                         selected_anntokens=annotation_list)

                # If this same was not there in the list already add an empty array to hold the annotations
                if camera_token not in image_data_dict:
                    image_data_dict[camera_token] = {'iID': camera_token,
                                                     'filePath': filename,
                                                     'annotations': []}

                # For each of the annotations in that sample
                for box in boxes:
                    corners = view_points(box.corners(), np.array(camera_intrinsic), normalize=True)[:2, :]
                    image_data_dict[camera_token]['annotations'].append({'name': box.name,
                                                                         'corners': corners.tolist()})
                # Get the next token
                token = sample['next']

        logger.info("Saving extracted image data dict")
        json_file_path = os.path.join(self.output_path, "camera_front_boxes.json")
        save_json(Path(json_file_path), image_data_dict)

# if __name__ == '__main__':

    #os.chdir(ROOT_DIR)
    # data_loader = DataLoader()
    # logger.info("Loading nuscenes data.")
    # nuscene_obj = data_loader.load_nuscenes_data()
    #
    # data_processor = nuscenesDataProcesser(nuscene_obj)
    # image_dict = data_processor.extract_nuscenes()
    #
    # logger.info("Saving extracted image data dict")
    # output_path = os.path.join(data_loader.config.data_process['output_data_path'],"camera_front_boxes.json")
    # save_json(Path(output_path), image_dict)
