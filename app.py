import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model from the pickle file
model_filename = "autoencoder_model.pkl"
with open(model_filename, "rb") as f:
    deep_autoencoder_model = pickle.load(f)

# Function to preprocess the input image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    input_image = np.array(image) / 255.0  # Normalize
    return input_image.flatten().reshape(1, 784)  # Flatten

# Function to display images
def display_results(input_image, decoded_image):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(input_image.reshape(28, 28), cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(decoded_image.reshape(28, 28), cmap='gray')
    axes[1].set_title('Decoded Image')
    axes[1].axis('off')
    
    st.pyplot(fig)

# Streamlit application layout
st.title("MNIST Autoencoder")
st.write("Upload an image of a handwritten digit (28x28 pixels) to see its encoding and decoding.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Load and preprocess the uploaded image
    input_image = preprocess_image(Image.open(uploaded_file))

    # Get the decoded output
    decoded_output = deep_autoencoder_model.predict(input_image)

    # Display results
    display_results(input_image, decoded_output)

