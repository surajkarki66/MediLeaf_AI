import bentoml
import rembg

from bentoml.io import Image, JSON

from MediLeaf_AI.utils.common import image_to_array, map_predictions_to_species_with_proba, add_white_background

BENTO_MODEL_TAG = "mobilenetv2:it2qcfa5wkmz44a4"

classifier_runner = bentoml.tensorflow.get(BENTO_MODEL_TAG).to_runner()

medileaf_service = bentoml.Service("medileaf_classifier", runners=[classifier_runner])

session = rembg.new_session("u2netp")


@medileaf_service.api(input=Image(), output=JSON())
async def classify(input_image):
    img = add_white_background(session, input_image, size=(1600, 1200))
    img_arr = image_to_array(img)
    result =  await classifier_runner.async_run(img_arr)
    prediction_response = map_predictions_to_species_with_proba(result, "./classes.json")
    
    return prediction_response

