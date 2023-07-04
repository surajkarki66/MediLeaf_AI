from MediLeaf_AI.config.configuration import ConfigurationManager
from MediLeaf_AI.components.evaluation import Evaluation
from MediLeaf_AI import logger


STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        prepare_base_config = config.get_prepare_base_model_config()
        training_config = config.get_training_config()
        evaluation = Evaluation(val_config, prepare_base_config, training_config)
        evaluation.evaluation()
        evaluation.save_score()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
