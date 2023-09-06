import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Register the KerasLayer from TensorFlow Hub
import tensorflow_hub as hub

# Function to load model with custom objects
def load_model_with_custom_objects(model_path):
    custom_objects = {"KerasLayer": hub.KerasLayer}
    return keras.models.load_model(model_path, custom_objects=custom_objects)

# Load the trained model with custom objects
model = load_model_with_custom_objects("model.h5")

# Define class labels
labels = ['motorcycle', 'Truck', 'Car', 'Bus']

# Create a Streamlit app
st.title("Vehicle Classification App")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Make predictions when a button is clicked
    if st.button("Classify"):
        # Preprocess the image
        image = Image.open(uploaded_image)
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = image[np.newaxis, ...]

        # Get model predictions
        result = model.predict(image)
        predicted_class_index = np.argmax(result)
        predicted_class_probability = result[0][predicted_class_index]
        predicted_class_label = labels[predicted_class_index]

        # Display the prediction
        st.write(f"Predicted Class: {predicted_class_label}")
        st.write(f"Predicted Probability: {predicted_class_probability:.2f}")

# Add an optional sidebar for additional controls or information
# st.sidebar.title("Sidebar Title")
# Add widgets to the sidebar
