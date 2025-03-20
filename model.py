from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from torchvision.models import EfficientNet_B0_Weights
import torch
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
from datetime import datetime

# ----------------------------
# U-Net Segmentation Model
# ----------------------------
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

        self.enc1 = double_conv(n_channels, 64)
        self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = double_conv(128, 64)

        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(torch.nn.functional.max_pool2d(enc1, 2))
        enc3 = self.enc3(torch.nn.functional.max_pool2d(enc2, 2))
        enc4 = self.enc4(torch.nn.functional.max_pool2d(enc3, 2))

        dec3 = self.dec3(torch.cat([self.up3(enc4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))

        return torch.sigmoid(self.out(dec1))

# ----------------------------
# Model Loading Functions
# ----------------------------
def load_model(model_class, model_path, device, num_classes=1):

    model = model_class(n_channels=3, n_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Segmentation model loaded successfully!")
    return model

# ----------------------------
# Image Preprocessing
# ----------------------------
def preprocess_image(image, transform):

    image = Image.open(image).convert("RGB")
    return transform(image).unsqueeze(0)

# ----------------------------
# Segmentation Function
# ----------------------------
def segment_image(image, model_seg, device):
    """Performs segmentation using U-Net."""
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    image_tensor = preprocess_image(image, transform).to(device)
    
    with torch.no_grad():
        output = model_seg(image_tensor)

    mask = torch.sigmoid(output).squeeze(0).cpu().numpy()  # Convert to numpy for visualization

    return image_tensor, torch.from_numpy(mask).to(device)

# ----------------------------
# Wound Area Extraction
# ----------------------------

def extract_wound_area(image_tensor, mask):

    # Debugging
    print(f"Mask shape: {mask.shape}")  
    print(f"Image shape: {image_tensor.shape}")  

    # Ensure image is in correct format [C, H, W]
    if len(image_tensor.shape) == 4:  
        image_tensor = image_tensor.squeeze(0)

    # Convert tensors to NumPy
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy() * 255  
    image_np = image_np.astype(np.uint8)  # Convert to uint8 for saving

    # Convert mask to NumPy and apply a stronger threshold
    mask_np = mask.squeeze(0).cpu().numpy()
    
    binary_mask = (mask_np > 0.7).astype(np.float32)  # Hard threshold to define wound region
    soft_mask = np.clip(mask_np, 0, 1)  # Preserve soft edges
    
    # Combine masks: use binary mask to define the region, but keep soft edges
    final_mask = binary_mask * soft_mask
    
    # Expand mask to 3 channels for RGB images
    mask_3channel = np.stack([final_mask] * 3, axis=-1)  

    # Apply masking
    wound_area = (image_np * mask_3channel).astype(np.uint8)

    # Save the extracted wound area
    wound_image_path = f"wound_area_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    Image.fromarray(wound_area).save(wound_image_path)

    return wound_image_path

# ----------------------------
# Classification Function (Handles Unknown Class)
# ----------------------------
def classify_wound(image_path, model_cls, device):

    # Define class names
    class_names = [4, 3, 2, 1, "unknown"]
    
    # Prepare the image with the same transformations used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load and transform the image
    image_tensor = preprocess_image(image_path, transform).to(device)
    
    # Get model prediction
    with torch.no_grad():
        logits = model_cls(image_tensor).squeeze(0)
        raw_logits = logits.cpu().numpy()
        print(f"Raw logits: {raw_logits}")

        # Apply temperature scaling to prevent extreme confidence
        temperature = 10  # Adjust this value if needed
        logits = logits / temperature

        # Clip logits to prevent extreme values
        logits = torch.clamp(logits, min=-50, max=50)

        # Compute softmax probabilities
        probabilities = F.softmax(logits, dim=0)
        final_prediction = torch.argmax(probabilities).item()
        confidence = probabilities[final_prediction].item()

        print(f"Final prediction: {class_names[final_prediction]}")
        print(f"Confidence: {confidence:.4f}")

        # If confidence is too high and logit range is extreme, classify as "unknown"
        logit_range = np.max(raw_logits) - np.min(raw_logits)
        if logit_range > 1000 or confidence > 0.99:
            return "unknown"

        return class_names[final_prediction]

# FastAPI Server
# ----------------------------
app = FastAPI()

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
segmentation_model_path = "50_after_wound_segmentation_model_2025-02-24 14_28_10.pth"
classification_model_path = "wound_classification_model_2025-03-05_13-56-58.pth"

model_seg = load_model(UNet, segmentation_model_path, device, num_classes=1)

model_cls = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model_cls.classifier[1] = nn.Linear(model_cls.classifier[1].in_features, 5)
print(model_cls.classifier[1])
model_cls.load_state_dict(torch.load(classification_model_path, map_location=device))
model_cls.to(device)
model_cls.eval()
print("Classification model loaded successfully!")

@app.get("/")
def read_root():
    return {"message": "Wound Segmentation & Classification API"}

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):

    file_path = f"temp_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Step 1: Predict segmentation mask
    image_tensor, mask = segment_image(file_path, model_seg, device)

    # Step 2: Extract wound area from original image
    wound_image_path = extract_wound_area(image_tensor.squeeze(0), mask)

    # Step 3: Classify wound type using extracted wound area
    wound_class = classify_wound(wound_image_path, model_cls, device)

    # Step 4: Clean up files
    os.remove(file_path)
    os.remove(wound_image_path)
    
    return JSONResponse(content={"wound_class": wound_class})
