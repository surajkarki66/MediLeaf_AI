import tensorflow as tf


model_path = "./artifacts/training/models/MobileNetV2"
output_path = "./artifacts/training/models/MobileNetV2.tflite"

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
tflite_model = converter.convert()

# Save the model.
with open(output_path, 'wb') as f:
  f.write(tflite_model)


interpreter = tf.lite.Interpreter(model_path = output_path)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input Shape:", input_details[0]['shape'])
print("Input Type:", input_details[0]['dtype'])
print("Output Shape:", output_details[0]['shape'])
print("Output Type:", output_details[0]['dtype'])
