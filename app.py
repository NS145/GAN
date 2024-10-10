import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

# Load the trained autoencoder model
model = joblib.load('deep_autoencoder_model.pkl')

# Function to normalize and reshape the input image
def preprocess_image(image):
    image = np.array(image, dtype=np.float32) / 255.0
    return image.reshape((1, 784))  # Reshape to the input shape of the model

# Streamlit UI
st.title("MNIST Deep Autoencoder")
st.write("Upload an image of a digit (0-9):")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = tf.keras.preprocessing.image.load_img(uploaded_file, color_mode='grayscale', target_size=(28, 28))
    image = tf.keras.preprocessing.image.img_to_array(image)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Predict using the autoencoder
    reconstructed_image = model.predict(processed_image)

    # Display the original and reconstructed images
    st.subheader("Original Image")
    st.image(image.reshape(28, 28), caption="Uploaded Image", use_column_width=True, clamp=True)

    st.subheader("Reconstructed Image")
    st.image(reconstructed_image.reshape(28, 28), caption="Reconstructed Image", use_column_width=True, clamp=True)

    # Optionally, show the encoded representation
    encoded_output = model.layers[1](processed_image)  # Get encoder output
    st.subheader("Encoded Output")
    st.write(encoded_output)
