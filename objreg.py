import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
import zipfile
import os

# Load categories
categories = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"]

# Function to load the model
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT

def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()
    return model

model = load_model()

# Function to make predictions
def make_prediction(img):
    img_preprocess = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

# Function to create image with bounding boxes
def create_image_with_bboxes(img, prediction):
    img_tensor = torch.tensor(img)
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"],
                                          labels=prediction["labels"],
                                          colors=["red" if label == "person" else "green" for label in prediction["labels"]], width=2)
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)
    return img_with_bboxes_np

# Function to handle dataset upload and extract images
def handle_dataset_upload(zip_file):
    images = []

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Extract the contents of the ZIP file to a temporary directory
        temp_dir = "temp_dataset"
        zip_ref.extractall(temp_dir)

        # Get a list of image files in the extracted directory
        image_files = [f for f in os.listdir(temp_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Load each image and append to the images list
        for image_file in image_files:
            image_path = os.path.join(temp_dir, image_file)
            image = Image.open(image_path)
            images.append(image)

    return images

# Dashboard
st.title("Object Detector ")

# Step 1: Upload ZIP Dataset or Single Image
upload_file = st.file_uploader(label="Upload Dataset (ZIP file) or Single Image:", type=["zip"])

# Check if a file is uploaded
if upload_file:
    if zipfile.is_zipfile(upload_file):
        # Handle dataset upload and extract images
        images = handle_dataset_upload(upload_file)

        # Display the extracted images and predictions
        for image in images:
            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Make predictions and display bounding boxes
            image = image.convert('RGB')
            prediction = make_prediction(image)
            img_with_bbox = create_image_with_bboxes(np.array(image).transpose(2, 0, 1), prediction)

            # Display the image with bounding boxes
            st.subheader("Image with Bounding Boxes")
            st.image(img_with_bbox, use_column_width=True)

            # Display predicted probabilities
            del prediction["boxes"]
            st.subheader("Predicted Probabilities")
            st.write(prediction)

    else:
        # Single Image Prediction
        uploaded_image = Image.open(upload_file)
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Make predictions and display bounding boxes
        prediction = make_prediction(uploaded_image)
        img_with_bbox = create_image_with_bboxes(np.array(uploaded_image).transpose(2, 0, 1), prediction)

        # Display the image with bounding boxes
        st.subheader("Image with Bounding Boxes")
        st.image(img_with_bbox, use_column_width=True)

        # Display predicted probabilities
        del prediction["boxes"]
        st.subheader("Predicted Probabilities")
        st.write(prediction)
