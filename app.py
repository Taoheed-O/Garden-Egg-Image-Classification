import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch.nn as nn


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



# Define function to make a prediction
def predict(image, model, class_names):
    image = preprocess_image(image)
    with torch.no_grad():
        outputs = model(image)  # Forward pass
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # Convert to probabilities
        top_prob, top_class = torch.max(probabilities, dim=0)  # Get the most probable class
    
    return class_names[top_class.item()], top_prob.item()  # Return class name and probability

# Define class names (adjust based on your model)
class_names = ["ripe", "rotten", "unripe"]  # Modify for your dataset

# Build Streamlit UI
st.title("Garden egg Image Classification")
st.write("Upload any garden egg image for classification")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Classify Image"):
        label, confidence = predict(image, model, class_names)
        st.success(f"Prediction: **{label}** with confidence **{confidence:.2f}**")

