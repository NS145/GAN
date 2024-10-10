# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained CNN model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.array(image) / 255.0  # Normalize
    return image.reshape(1, 28, 28, 1)  # Reshape for the model

# Streamlit UI
st.title("MNIST Digit Classification with CNN")
st.write("Upload an image of a handwritten digit (0-9):")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image)

    # Make a prediction
    predictions = model.predict(processed_image)
    predicted_digit = np.argmax(predictions)

    # Display the uploaded image and the prediction
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.subheader(f"Predicted Digit: {predicted_digit}")
