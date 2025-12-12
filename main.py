from fastapi import FastAPI
import torch
from backend.train_model import CheXNetModel  # Import model class

app = FastAPI()

# Load model (trained weights should exist in models/)
model = CheXNetModel(num_classes=14)
model.load_state_dict(torch.load("backend/models/chexnet_best.pth", map_location="cpu"))
model.eval()

@app.get("/")
def home():
    return {"message": "API is running!"}

@app.get("/predict")
def predict():
    # Dummy prediction example
    sample_input = torch.randn(1, 3, 224, 224)
    output = model(sample_input)
    return {"prediction": output.tolist()}
