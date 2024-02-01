import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
import zipfile
import os

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

# Function to load the model
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT

def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()
    return model

model = load_model()

# Function to make predictions
def make_prediction(img):
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

# Function to create image with bounding boxes
def create_image_with_bboxes(img, prediction):
    img_tensor = torch.tensor(img)
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"],
                                          labels=prediction["labels"],
                                          colors=["red" if label == "person" else "green" for label in
                                                  prediction["labels"]], width=2)
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)
    return img_with_bboxes_np

# Dashboard
st.markdown("<h1 style='text-align: center;'>Object Detector Model</h1>", unsafe_allow_html=True)

# Step 1: Upload ZIP Dataset
upload_dataset = st.file_uploader(label="Upload Dataset (ZIP file format):", type=["zip"])

if upload_dataset:
    # Extract images directly within the Streamlit dashboard
    with zipfile.ZipFile(upload_dataset, 'r') as zip_ref:
        # Extract the contents of the ZIP file to a temporary directory
        temp_dir = "temp_images"
        zip_ref.extractall(temp_dir)

        # Get a list of image files in the extracted directory
        image_files = [f for f in os.listdir(temp_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Display each image with object detection results
        for image_file in image_files:
            image_path = os.path.join(temp_dir, image_file)
            image = Image.open(image_path)

            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Make predictions and display bounding boxes
            image = image.convert('RGB')
            prediction = make_prediction(image)
            img_with_bbox = create_image_with_bboxes(np.array(image).transpose(2, 0, 1), prediction)

            # Display the image with bounding boxes
            st.image(img_with_bbox, caption="Image with Bounding Boxes", use_column_width=True)

            # Display predicted probabilities
            del prediction["boxes"]
            st.header("Predicted Probabilities")
            st.write(prediction, style={"labels": {"font-size": "20px"}})
