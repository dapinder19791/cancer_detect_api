# Clinical Image Assist - FastAPI Backend

AI-powered chest X-ray analysis backend using CheXNet-based deep learning model.

## Features

- **Multi-class Classification**: Detects 14 different chest X-ray pathologies
- **Grad-CAM Visualization**: Explainable AI heatmaps showing model focus areas
- **Clinical Report Generation**: Optional Azure OpenAI integration for professional reports
- **Supabase Integration**: Stores analysis results and metadata

## Installation

1. **Create virtual environment**:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your credentials
```

## Configuration

Copy the Supabase credentials from your frontend `.env` file:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
```

### Optional: Azure OpenAI Setup

For AI-generated clinical reports, add:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your_azure_openai_key
```

## Running the Server

```bash
python app.py
```

Server runs on `http://localhost:8000`

## API Endpoints

### POST /api/predict

Upload X-ray image for analysis.

**Request**:
- `file`: X-ray image file (multipart/form-data)
- `analysis_id`: UUID from database (form field)

**Response**:
```json
{
  "success": true,
  "analysis_id": "uuid",
  "prediction": {
    "class": "Pneumonia",
    "confidence": 87.5,
    "all_predictions": {
      "Pneumonia": 87.5,
      "Infiltration": 45.2,
      ...
    }
  },
  "processing_time_ms": 1234
}
```

## Model Details

- **Architecture**: DenseNet-121
- **Training**: Transfer learning from ImageNet
- **Classes**: 14 pathology types from NIH Chest X-Ray dataset
- **Input**: 224x224 RGB images
- **Output**: Multi-label probabilities

### Detected Conditions

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural Thickening
14. Hernia

## Training Your Own Model

To replace the demo model with a trained version:

1. **Dataset**: Download NIH Chest X-Ray14 or similar dataset
2. **Training**: Use `train_model.py` (see training notebook)
3. **Save weights**: Export model as `chexnet_weights.pth`
4. **Update**: Place weights in `backend/models/` directory

## Development

### Run with auto-reload:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Testing:
```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@sample_xray.jpg" \
  -F "analysis_id=your-uuid-here"
```

## Production Deployment

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

### Azure Container Apps
```bash
az containerapp create \
  --name clinical-image-assist \
  --resource-group myResourceGroup \
  --environment myEnvironment \
  --image myregistry.azurecr.io/clinical-assist:latest \
  --target-port 8000
```

## Performance

- **CPU Inference**: ~1-2 seconds per image
- **GPU Inference**: <200ms per image
- **Memory**: ~2GB for model

## Security & Compliance

⚠️ **Important**: This is a demonstration project.

- Not FDA approved
- Not for clinical diagnosis
- Always verify with qualified radiologists
- HIPAA compliance required for production use
- Implement proper authentication for production

## Troubleshooting

**CORS errors**: Check frontend URL in `allow_origins`

**Model not loading**: Ensure PyTorch is installed correctly

**Supabase errors**: Verify credentials and database schema

**Azure OpenAI errors**: Check endpoint and API key configuration

## License

MIT License - For educational and research purposes only.
