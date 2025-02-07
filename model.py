from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
import random
import torch.nn as nn
import numpy as np
from PIL import Image
import os

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Encoder (downsampling)
        self.enc1 = double_conv(n_channels, 64)
        self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)

        # Decoder (upsampling)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = double_conv(128, 64)

        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.functional.max_pool2d(enc1, 2))
        enc3 = self.enc3(nn.functional.max_pool2d(enc2, 2))
        enc4 = self.enc4(nn.functional.max_pool2d(enc3, 2))

        # Decoder
        dec3 = self.dec3(torch.cat([self.up3(enc4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))

        return self.out(dec1)

def load_model(model_path, device):
    model = UNet(n_channels=3, n_classes=1)  # Make sure this matches your trained model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_woundarea(image, mask):

    # Convert image to PyTorch tensor and move to same device as mask
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        image = image.to(mask.device)  # Move to same device as mask

    mask_expanded = mask.squeeze(0)  # Remove batch dimension if present
    if len(mask_expanded.shape) == 2:  # If single channel
        mask_expanded = mask_expanded.unsqueeze(0)  # Add channel dimension

    if image.shape[0] == 3:  # RGB image
        mask_expanded = mask_expanded.repeat(3, 1, 1)  # [3, H, W] to match the image channels

    wound_area = image * mask_expanded  # Element-wise multiply image and mask
    wound_area_np = wound_area.permute(1, 2, 0).cpu().numpy()  # Convert for plotting
    return wound_area_np


#model predict part
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def postprocess_mask(mask):
    mask = mask.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask

def processdata(model, image_path, mask_path, device):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    image = preprocess_image(image_path, transform)
    image = image.to(device)

    actual_mask = Image.open(mask_path).convert("L")
    actual_mask = transform(actual_mask).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        predicted_mask = torch.sigmoid(output)

    return (
        image.squeeze().cpu().permute(1, 2, 0).numpy(),
        actual_mask,
        predicted_mask
    )

def segmentation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    file_name = "fusc_0049.png"
    model_path = "/content/drive/My Drive/Colab Notebooks/Wound Model/100_wound_segmentation_model_2024-10-12 12_24_22.pth"
    # test_image_path = "/content/drive/My Drive/Colab Notebooks/Wound Model/data/test_images/" + file_name
    test_image_path = "/content/drive/My Drive/Colab Notebooks/Wound Model/data/classification data/Class 1 - รอยแดง/0a9318e35fc7a644b06b54d5b8699c869cf1d57_0.jpg"
    actual_mask_path = "/content/drive/My Drive/Colab Notebooks/Wound Model/data/test_masks/" + file_name

    # Define path to save the extracted wound area
    wound_area_save_path = f"/content/drive/My Drive/Colab Notebooks/Wound Model/data/classification data/after_segment/Class_1/{file_name}"

    model = load_model(model_path, device)
    image, actual_mask, predicted_mask = processdata(model, test_image_path, actual_mask_path, device)

    wound_area = get_woundarea(image, predicted_mask)

    if wound_area_save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(wound_area_save_path), exist_ok=True)

        # Save only the extracted wound area
        wound_area_image = Image.fromarray((wound_area * 255).astype(np.uint8))
        wound_area_image.save(wound_area_save_path)

app = FastAPI()

#test Fast Api

@app.get("/")
def read_root():
    return {"Hello World"}

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    # Read the file contents
    file_content = await file.read()
    
    # Process the file content here (e.g., run a prediction model)
    # For now, we'll just return a message with the filename and file size+
    file_size = len(file_content)
    print("file size :",file_size)
    mockwound_state = random.randint(1, 4)

    segmentation()
    
    # Return a response
    return JSONResponse(content={
        "wound_state": mockwound_state,
    })

# Load your trained model
# model = torch.load("model.pth")
# model.eval()  # Set model to evaluation mode

# @app.post("/predict/")
# async def predict(input_data: dict):
#     # Convert input_data to tensor (example assumes input_data is a dict with 'data' key)
#     input_tensor = torch.tensor(input_data['data'])
    
#     # Perform model prediction
#     with torch.no_grad():
#         prediction = model(input_tensor)
    
#     # Convert prediction to a Python list (to be JSON serializable)
#     return {"prediction": prediction.tolist()}

