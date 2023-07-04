from MediLeaf_AI.config.configuration import ConfigurationManager
from MediLeaf_AI.components.deployment import Deployment
from MediLeaf_AI import logger




STAGE_NAME = "Deployment stage"


class DeploymentPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        deployment_config = config.get_deployment_config()
        prepare_base_config = config.get_prepare_base_model_config()
        deployment = Deployment(deployment_config, prepare_base_config)
        deployment.deploy()



if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DeploymentPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
            