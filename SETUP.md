# Backend Setup Instructions

The Python environment in this container doesn't have pip installed. Here's how to set up the backend on your local machine:

## Local Setup (Recommended)

### 1. Install Python 3.10+

Make sure you have Python 3.10 or higher installed:
```bash
python3 --version
```

### 2. Create Virtual Environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- FastAPI
- Uvicorn
- PyTorch & TorchVision
- OpenCV
- Supabase client
- And other required packages

### 4. Run the Server

```bash
python app.py
```

The server will start on `http://localhost:8000`

### 5. Test the API

```bash
curl http://localhost:8000
```

You should see:
```json
{
  "message": "Clinical Image Assist API",
  "version": "1.0",
  "model": "CheXNet-based DenseNet121",
  "endpoints": ["/api/predict"]
}
```

## Docker Setup (Alternative)

If you prefer Docker:

```bash
cd backend
docker build -t clinical-assist-backend .
docker run -p 8000:8000 --env-file .env clinical-assist-backend
```

## Quick Test Without Backend

The frontend will work for UI demonstration, but will show an error when trying to analyze images without the backend running.

To test the complete pipeline, you MUST have the FastAPI backend running.

## Troubleshooting

**Import errors**: Make sure all dependencies are installed
**Port already in use**: Change port in app.py or kill the process using port 8000
**CORS errors**: Check that the backend URL matches in frontend .env
