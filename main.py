from visualInsights import logger
from visualInsights.pipeline.data_loader import DataLoader, nuscenesDataProcesser

STAGE_NAME = "Data Loading & Processing Stage"
logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
try:
    data_loader = DataLoader()
    logger.info("Loading nuscenes data.")
    nuscene_obj = data_loader.load_nuscenes_data()

    data_processor = nuscenesDataProcesser(nuscene_obj)
    image_dict = data_processor.extract_nuscenes()

    logger.info("Generating class distribution data from nuscenes")
    data_processor.get_class_distribution(image_dict)

except Exception as e:
    logger.exception(e)
    raise e
