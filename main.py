import os
from pathlib import  Path
from visualInsights.constants import *
from visualInsights import logger
from visualInsights.utils.utils import read_yaml, load_json
from visualInsights.pipeline.data_loader import DataLoader, nuscenesDataExtractor
from visualInsights.pipeline.data_process import DataProcessor

#flags to control which stage to run
run_data_loader = True
run_data_processor = True

config = read_yaml(CONFIG_FILE_PATH)
output_path = config.data_output['output_data_path']
nuscenes_version = config.data_loader['nuscenes_version']

if run_data_loader:
    STAGE_NAME = "Stage 1. Data Loading & Extraction"
    logger.info(f">>>>>> {STAGE_NAME} Started <<<<<<")
    try:
        data_loader = DataLoader()
        logger.info("Loading nuscenes data.")
        nuscene_obj = data_loader.load_nuscenes_data()

        logger.info("Extracting annotations..")
        data_extractor = nuscenesDataExtractor(nuscene_obj)
        data_extractor.extract_nuscenes()
        logger.info(f">>>>>> {STAGE_NAME} Completed! <<<<<<")

    except Exception as e:
        logger.exception(e)
        raise e

if run_data_processor:
    STAGE_NAME = "Stage 2. Data Processing"
    logger.info(f">>>>>> {STAGE_NAME} Started <<<<<<")
    try:
        json_filename = os.path.join(output_path, nuscenes_version + "_camera_boxes.json")
        data = load_json(Path(json_filename))

        logger.info("Generating class distribution data from nuscenes")
        data_processor = DataProcessor()
        data_processor.get_class_distribution(data)
        logger.info(f">>>>>> {STAGE_NAME} Completed! <<<<<<")

    except Exception as e:
        logger.exception(e)
        raise e
