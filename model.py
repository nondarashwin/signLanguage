from types import FunctionType

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load the model
loaded_model = tf.keras.models.load_model("./model.h5")
img_height = 288
img_width = 288

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
               'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def build_model(loaded):
    keras_layer = hub.KerasLayer(loaded, trainable=True)
    model = tf.keras.Model(keras_layer)
    return model


def read_image(file_name):
    image = tf.io.read_file(file_name)
    image = tf.io.decode_jpeg(image, channels=3)
    #image = tf.image.convert_image_dtype(image, tf.float32)
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    image = tf.image.resize_with_pad(image, target_height=img_height, target_width=img_width)
    return np.expand_dims(normalization_layer(image), axis=0)


def preprocess_image(frame):
    # Resize the frame to match the input size of your model
    resized_frame = cv2.resize(frame, (img_width, img_height))
    resized_frame = cv2.blur(resized_frame, (3, 3))
    # Normalize the pixel values to be in the range [0, 1]
    normalized_frame = resized_frame / 255.0
    # Expand the dimensions to match the shape expected by the model
    input_data = np.expand_dims(normalized_frame, axis=0)
    return input_data


# Now you can use this loaded_model for inference or further training
def predict(img):
    global loc1
    image = read_image(img)
    print(image.shape)
    #image = tf.image.resize_with_pad(image, target_height=288, target_width=288)
    predictions = loaded_model.predict(image)
    max_value = max(predictions[0])
    loc = np.argmax(loaded_model.predict(image), axis=1)
    print(predictions[0])
    print(max_value)
    print(loc[0])
    output_layer = loaded_model.get_layer("dense_1")

    # Get the class name of the output layer
    class_name = output_layer.__class__.__name__

    print("Output layer class name:", class_name)

    return class_names[loc[0]]
