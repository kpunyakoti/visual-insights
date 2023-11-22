import os
from visualInsights import logger
from visualInsights.pipeline.data_loader import DataLoader, nuscenesDataProcesser

STAGE_NAME = "Data Loading & Processing Stage"
logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
try:
    data_loader = DataLoader()
    logger.info("Loading nuscenes data.")
    nuscene_obj = data_loader.load_nuscenes_data()

    data_processor = nuscenesDataProcesser(nuscene_obj)
    data_processor.extract_nuscenes()

except Exception as e:
    logger.exception(e)
    raise e

