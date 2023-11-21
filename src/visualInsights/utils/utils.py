"""Utils for visual-insights data processing pipeline."""

import pandas as pd
from nuscenes.nuscenes import NuScenes

# Define Data Loader
class DataLoader():

    def __init__(self,  file_path, version, source_type = "nuscenes"):
        self.version = version
        self.data_path = file_path

    def load_nuscenes_data(self):
        logging
        nusc = NuScenes(version=self.version, dataroot=self.data_path, verbose=True)
