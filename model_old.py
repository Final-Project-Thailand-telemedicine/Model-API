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
    """Loads a trained model from a file."""
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
    """Preprocesses an image for model input."""
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
    
    mask = output.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255  # Binary mask

    return image_tensor, mask

# ----------------------------
# Wound Area Extraction
# ----------------------------
def extract_wound_area(image_tensor, mask):
    """Extracts the wound area using the segmentation mask."""
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(image_tensor.device)
    wound_area = image_tensor * mask_tensor
    wound_area_np = wound_area.squeeze().cpu().permute(1, 2, 0).numpy()
    
    wound_image = Image.fromarray((wound_area_np * 255).astype(np.uint8))
    wound_image_path = "temp_wound.png"
    wound_image.save(wound_image_path)
    
    return wound_image_path

# ----------------------------
# Classification Function (Handles Unknown Class)
# ----------------------------
def classify_wound(image_path, model_cls, device, threshold=0.5):
    """Classifies a wound image and returns 'unknown' if confidence is too low."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_tensor = preprocess_image(image_path, transform).to(device)
    
    with torch.no_grad():
        outputs = model_cls(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)

    print(f"Predicted wound class: {preds.item()}, Confidence: {confidence.item():.4f}")

    # Return "unknown" if confidence is too low
    print(f"confidence,preds: {confidence.item()},{preds.item()}")
    if preds.item() < threshold:
        return "unknown"

    return int(preds.item()) + 1  # Convert class index (0-3) to (1-4)

# ----------------------------
# FastAPI Server
# ----------------------------
app = FastAPI()

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
segmentation_model_path = "50_after_wound_segmentation_model_2025-02-24 14_28_10.pth"
classification_model_path = "wound_classification_model_2025-02-27_06-11-59.pth"

model_seg = load_model(UNet, segmentation_model_path, device, num_classes=1)

model_cls = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model_cls.classifier[1] = nn.Linear(model_cls.classifier[1].in_features, 4)
model_cls.load_state_dict(torch.load(classification_model_path, map_location=device))
model_cls.to(device)
model_cls.eval()
print("Classification model loaded successfully!")

@app.get("/")
def read_root():
    return {"message": "Wound Segmentation & Classification API"}

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    """Handles image upload, segmentation, and classification."""
    file_path = f"temp_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    image_tensor, mask = segment_image(file_path, model_seg, device)
    wound_image_path = extract_wound_area(image_tensor, mask)
    wound_class = classify_wound(wound_image_path, model_cls, device)
    
    os.remove(file_path)
    os.remove(wound_image_path)
    
    return JSONResponse(content={"wound_class": wound_class})
