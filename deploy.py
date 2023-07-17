from MediLeaf_AI import logger
from MediLeaf_AI.pipeline.stage_05_deployment import DeploymentPipeline


STAGE_NAME = "Deployment stage"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DeploymentPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
        

