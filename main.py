from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
import io
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_names = ['fake', 'real']

try:
    print("NumPy version:", np.__version__)
except Exception as e:
    print("NumPy is not available:", str(e))

model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load('model_resnet18.pth', map_location=device))
model = model.to(device)
model.eval()

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def transform_image_to_fft(image: Image.Image) -> np.ndarray:
    img = np.array(image.convert('L'))
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_spectrum = np.uint8(magnitude_spectrum)
    return Image.fromarray(magnitude_spectrum).convert("RGB")

def save_spectrum_image(image: Image.Image, filename: str):
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    image.save(filepath)
    return filepath

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    print("Received file:", file.filename)
    try:
        image = Image.open(io.BytesIO(await file.read()))
    except Exception as e:
        print("Error reading image:", e)
        return JSONResponse(content={"error": str(e)}, status_code=400)
    
    fft_image = transform_image_to_fft(image)
    spectrum_filepath = save_spectrum_image(fft_image, file.filename)
    print("Spectrum image saved to:", spectrum_filepath)

    try:
        fft_image = data_transform(fft_image).unsqueeze(0).to(device)
    except Exception as e:
        print("Error transforming image:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    with torch.no_grad():
        outputs = model(fft_image)
        probabilities = F.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)
        predicted_class = class_names[preds[0]]
        confidence_percentage = confidence[0].item() * 100
    
    return JSONResponse(content={
        "prediction": predicted_class,
        "confidence": confidence_percentage,
        "spectrum_image_path": spectrum_filepath
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
