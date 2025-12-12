from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import numpy as np
import cv2
from datetime import datetime
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import Optional
import traceback

load_dotenv()

app = FastAPI(title="Clinical Image Assist API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]

class CheXNetModel(nn.Module):
    def __init__(self, num_classes=14):
        super(CheXNetModel, self).__init__()
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.densenet.features(x)
        out = torch.relu(features)
        out = torch.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.densenet.classifier(out)
        return out, features

print("Loading model...")
model = CheXNetModel(num_classes=len(CLASSES))
model.eval()
print("Model loaded successfully")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def generate_gradcam(model, image_tensor, target_class_idx=None):
    try:
        features_blobs = []

        def hook_feature(module, input, output):
            features_blobs.append(output.data.cpu().numpy())

        model.densenet.features.register_forward_hook(hook_feature)

        logits, _ = model(image_tensor)

        if target_class_idx is None:
            target_class_idx = logits.argmax(dim=1).item()

        model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0][target_class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=True)

        features = features_blobs[0][0]

        weights = np.mean(features, axis=(1, 2))
        cam = np.zeros(features.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * features[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)

        return cam
    except Exception as e:
        print(f"Grad-CAM generation error: {e}")
        return None

def apply_heatmap(image, cam):
    try:
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        img_np = np.array(image.resize((224, 224)))
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

        superimposed = heatmap * 0.4 + img_np * 0.6
        superimposed = np.uint8(superimposed)

        return superimposed
    except Exception as e:
        print(f"Heatmap application error: {e}")
        return None

async def generate_clinical_report(prediction_class: str, confidence: float, all_predictions: dict) -> Optional[str]:
    try:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_KEY")

        if not azure_endpoint or not azure_key:
            return None

        from openai import AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version="2024-02-01"
        )

        top_findings = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)[:5]
        findings_text = "\n".join([f"- {name}: {prob:.1f}%" for name, prob in top_findings])

        prompt = f"""You are an experienced radiologist assistant. Based on the AI analysis of a chest X-ray, generate a professional clinical report.

Primary Finding: {prediction_class} (Confidence: {confidence:.1f}%)

Top Findings:
{findings_text}

Please provide a structured clinical report including:
1. FINDINGS: Brief description of detected abnormalities
2. IMPRESSION: Clinical interpretation
3. RECOMMENDATIONS: Suggested follow-up or additional imaging if needed

Keep the report concise, professional, and clinically appropriate. Remember this is AI-assisted and should be verified by a radiologist."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert radiologist assistant providing clinical interpretations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Clinical report generation error: {e}")
        return None

@app.get("/")
async def root():
    return {
        "message": "Clinical Image Assist API",
        "version": "1.0",
        "model": "CheXNet-based DenseNet121",
        "endpoints": ["/api/predict"]
    }

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    analysis_id: str = Form(...)
):
    start_time = datetime.now()

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        original_size = image.size

        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs, _ = model(image_tensor)
            probabilities = outputs.squeeze().numpy() * 100

        predictions_dict = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}

        top_class_idx = probabilities.argmax()
        prediction_class = CLASSES[top_class_idx]
        confidence = float(probabilities[top_class_idx])

        cam = generate_gradcam(model, image_tensor, top_class_idx)
        heatmap_base64 = None

        if cam is not None:
            heatmap_img = apply_heatmap(image, cam)
            if heatmap_img is not None:
                _, buffer = cv2.imencode('.png', cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))
                heatmap_base64 = f"data:image/png;base64,{base64.b64encode(buffer).decode()}"

        clinical_report = await generate_clinical_report(
            prediction_class,
            confidence,
            predictions_dict
        )

        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

        update_data = {
            "prediction_class": prediction_class,
            "confidence_score": confidence,
            "predictions_json": predictions_dict,
            "heatmap_url": heatmap_base64,
            "clinical_report": clinical_report,
            "status": "completed",
            "updated_at": datetime.now().isoformat()
        }

        supabase.table("xray_analyses").update(update_data).eq("id", analysis_id).execute()

        metadata = {
            "analysis_id": analysis_id,
            "model_version": "CheXNet-DenseNet121-v1.0",
            "processing_time_ms": processing_time,
            "image_dimensions": {"width": original_size[0], "height": original_size[1]},
            "preprocessing_params": {
                "resize": "224x224",
                "normalization": "ImageNet",
                "grayscale_to_rgb": True
            }
        }

        supabase.table("analysis_metadata").insert(metadata).execute()

        return JSONResponse({
            "success": True,
            "analysis_id": analysis_id,
            "prediction": {
                "class": prediction_class,
                "confidence": confidence,
                "all_predictions": predictions_dict
            },
            "processing_time_ms": processing_time
        })

    except Exception as e:
        error_message = str(e)
        print(f"Prediction error: {error_message}")
        print(traceback.format_exc())

        try:
            supabase.table("xray_analyses").update({
                "status": "failed",
                "error_message": error_message,
                "updated_at": datetime.now().isoformat()
            }).eq("id", analysis_id).execute()
        except:
            pass

        raise HTTPException(status_code=500, detail=error_message)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
