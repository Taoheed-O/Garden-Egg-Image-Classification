import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch.nn as nn
import numpy as np

model_path =  "model.pth"
# Load the trained PyTorch model
@st.cache_resource


# Define the model architecture (must match the saved model)
class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # self.conv_block_3 = nn.Sequential(
        #     nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        # x = self.conv_block_1(x)
        # # print(x.shape)
        # x = self.conv_block_2(x)
        # # print(x.shape)
        # x = self.classifier(x)
        # # print(x.shape)
        # return x
        return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion

# Create model instance
model = TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                  hidden_units=256, 
                  output_shape=3)  # Ensure the number of classes matches

# Load the state dictionary
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

# Set model to evaluation mode
model.eval()



# Define the image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


# Prediction function for multiple images
# Prediction function
def predict_image(model, image):
    img_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)  # Get class index
    return predicted.item()


# Grad-CAM Implementation
def generate_gradcam(model, image, target_layer):
    model.eval()
    img_tensor = preprocess_image(image)
    img_tensor.requires_grad = True

    # Forward pass
    output = model(img_tensor)
    pred_class = output.argmax().item()

    # Backward pass
    model.zero_grad()
    output[0, pred_class].backward()

    # Get gradients of the target layer
    gradients = target_layer.weight.grad
    activations = target_layer.weight.data

    # Compute Grad-CAM
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * activations, dim=1).squeeze().detach().numpy()
    
    cam = np.maximum(cam, 0)  # Apply ReLU
    cam = transforms.Resize(64, 64)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)  # Normalize to 0-1
    return cam


# Select the target layer for Grad-CAM
target_layer = model.conv_block_2[-3]


# Streamlit UI
st.title("Garden Egg Image Classification")
st.write("Upload images or use your webcam to get predictions.")


# Upload multiple images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Webcam image capture
captured_image = st.camera_input("Take a picture")

# Class names (replace with actual labels)
class_names = ["ripe", "rotten", "unripe"]  

# Process uploaded images
if uploaded_files:
    images = [Image.open(file).convert("RGB") for file in uploaded_files]  # Convert to RGB
    predictions = [predict_image(model, img) for img in images]  # Predict each image

    # Display images with predictions
    st.subheader("Uploaded Images Predictions")
    cols = st.columns(len(images))  # Arrange images in a row
    for idx, col in enumerate(cols):
        col.image(images[idx], caption=f"Predicted: {class_names[predictions[idx]]}", use_container_width=True)

        # Generate Grad-CAM heatmap
        heatmap = generate_gradcam(model, images[idx], target_layer)
        col.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True)

# Process webcam image
if captured_image:
    webcam_image = Image.open(captured_image).convert("RGB")  # Convert to RGB
    prediction = predict_image(model, webcam_image)  # Predict

    # Display webcam image and prediction
    st.subheader("Webcam Image Prediction")
    st.image(webcam_image, caption=f"Predicted: {class_names[prediction]}", use_container_width=True)

    # Generate Grad-CAM heatmap
    heatmap = generate_gradcam(model, webcam_image, target_layer)
    st.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True)
