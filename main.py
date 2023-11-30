from visualInsights.constants import *
from visualInsights import logger
from visualInsights.pipeline.data_loader import DataLoader, nuscenesDataExtractor
from visualInsights.pipeline.data_process import DataProcessor
from visualInsights.pipeline.generate_feature_vectors import generateFeatureVectorDB
from visualInsights.pipeline.compute_image_similarities import computeImageSimilarity

#flags to control which stage to run
run_meta_data_extractor = False
generate_statistics = False
generate_feature_vector_db = False
compute_image_similarities = True
fetch_similar_images = False

if run_meta_data_extractor:
    STAGE_NAME = "Stage 1. Load & Extract Meta Data"
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

if generate_statistics:
    STAGE_NAME = "Stage 2. Generate Statistics"
    logger.info(f">>>>>> {STAGE_NAME} Started <<<<<<")
    try:
        logger.info("Generating class distribution data from nuscenes.")
        data_processor = DataProcessor()
        data_processor.get_class_distribution()
        logger.info("Fetching model performance scores.")
        data_processor.get_performance_scores()
        logger.info(f">>>>>> {STAGE_NAME} Completed! <<<<<<")

    except Exception as e:
        logger.exception(e)
        raise e

if generate_feature_vector_db:
    STAGE_NAME = "Stage 3. Generate Feature Vecotrs"
    logger.info(f">>>>>> {STAGE_NAME} Started <<<<<<")
    try:
        logger.info("Generating feature vectors for all images.")
        feature_vector = generateFeatureVectorDB()
        feature_vector.generate_feature_vectors()
        logger.info(f">>>>>> {STAGE_NAME} Completed! <<<<<<")

    except Exception as e:
        logger.exception(e)
        raise e

if compute_image_similarities:
    STAGE_NAME = "Stage 4. Compute Image Similarities"
    logger.info(f">>>>>> {STAGE_NAME} Started <<<<<<")
    try:
        logger.info("Computing image similarity scores")
        img_similarity = computeImageSimilarity()
        img_similarity.compute_top_n_similar_images()
        logger.info(f">>>>>> {STAGE_NAME} Completed! <<<<<<")

    except Exception as e:
        logger.exception(e)
        raise e
