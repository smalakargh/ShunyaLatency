import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("kws_model.h5")
X_train = np.load("X_data.npy")

# Generator function to calibrate INT8 quantization range using representative data
def representative_dataset_gen():
        for i in range(min(50, len(X_train))):
            # Ensure sample is 4D: [1, Height, Width, Channels]
            sample = np.expand_dims(X_train[i], axis=0).astype(np.float32)
            if len(sample.shape) == 3:  # If missing channel dimension
                sample = np.expand_dims(sample, axis=-1)
            yield [sample]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Successfully generated quantized model.tflite")