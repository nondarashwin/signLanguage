import tensorflow as tf

# Load the model
loaded_model = tf.saved_model.load("./model.keras")
img_height = 288
img_width = 288


# Now you can use this loaded_model for inference or further training
def predict(img):
    image = tf.io.decode_jpeg(img, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, target_height=288, target_width=288)
    # image = tf.image.resize_with_pad(img, target_height=288, target_width=288)
    predictions = loaded_model.predict(image)
    return predictions
