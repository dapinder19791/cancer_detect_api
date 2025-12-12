# Deployment Guide - Clinical Image Assist

Complete deployment guide for production environments.

## Prerequisites

- Docker installed
- Azure account (for Azure deployment)
- Supabase project configured
- Domain name (optional)

## Local Development

```bash
# Frontend
npm run dev

# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Docker Deployment

### Backend Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
```

### Frontend Dockerfile

```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

RUN npm run build

RUN npm install -g serve

EXPOSE 3000

CMD ["serve", "-s", "dist", "-l", "3000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_ANON_KEY=${SUPABASE_ANON_KEY}
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - AZURE_OPENAI_KEY=${AZURE_OPENAI_KEY}
    volumes:
      - ./backend:/app
    restart: unless-stopped

  frontend:
    build: .
    ports:
      - "3000:3000"
    environment:
      - VITE_SUPABASE_URL=${VITE_SUPABASE_URL}
      - VITE_SUPABASE_ANON_KEY=${VITE_SUPABASE_ANON_KEY}
      - VITE_API_URL=http://backend:8000
    depends_on:
      - backend
    restart: unless-stopped
```

## Azure Container Apps Deployment

### 1. Create Azure Container Registry

```bash
az group create --name clinical-assist-rg --location eastus

az acr create \
  --resource-group clinical-assist-rg \
  --name clinicalassistacr \
  --sku Basic

az acr login --name clinicalassistacr
```

### 2. Build and Push Images

```bash
# Backend
cd backend
docker build -t clinicalassistacr.azurecr.io/backend:latest .
docker push clinicalassistacr.azurecr.io/backend:latest

# Frontend
cd ..
docker build -t clinicalassistacr.azurecr.io/frontend:latest .
docker push clinicalassistacr.azurecr.io/frontend:latest
```

### 3. Create Container Apps Environment

```bash
az containerapp env create \
  --name clinical-assist-env \
  --resource-group clinical-assist-rg \
  --location eastus
```

### 4. Deploy Backend

```bash
az containerapp create \
  --name clinical-assist-backend \
  --resource-group clinical-assist-rg \
  --environment clinical-assist-env \
  --image clinicalassistacr.azurecr.io/backend:latest \
  --target-port 8000 \
  --ingress external \
  --registry-server clinicalassistacr.azurecr.io \
  --cpu 2.0 \
  --memory 4.0Gi \
  --min-replicas 1 \
  --max-replicas 5 \
  --env-vars \
    SUPABASE_URL=secretref:supabase-url \
    SUPABASE_ANON_KEY=secretref:supabase-key \
    AZURE_OPENAI_ENDPOINT=secretref:openai-endpoint \
    AZURE_OPENAI_KEY=secretref:openai-key
```

### 5. Deploy Frontend

```bash
az containerapp create \
  --name clinical-assist-frontend \
  --resource-group clinical-assist-rg \
  --environment clinical-assist-env \
  --image clinicalassistacr.azurecr.io/frontend:latest \
  --target-port 3000 \
  --ingress external \
  --registry-server clinicalassistacr.azurecr.io \
  --cpu 0.5 \
  --memory 1.0Gi \
  --env-vars \
    VITE_API_URL=https://clinical-assist-backend.azurecontainerapps.io \
    VITE_SUPABASE_URL=secretref:supabase-url \
    VITE_SUPABASE_ANON_KEY=secretref:supabase-key
```

### 6. Configure Secrets

```bash
az containerapp secret set \
  --name clinical-assist-backend \
  --resource-group clinical-assist-rg \
  --secrets \
    supabase-url=YOUR_SUPABASE_URL \
    supabase-key=YOUR_SUPABASE_KEY \
    openai-endpoint=YOUR_OPENAI_ENDPOINT \
    openai-key=YOUR_OPENAI_KEY
```

## Alternative: Azure Static Web Apps + Functions

### Frontend (Static Web App)

```bash
az staticwebapp create \
  --name clinical-assist \
  --resource-group clinical-assist-rg \
  --location eastus \
  --sku Standard
```

Deploy via GitHub Actions (automatic on push).

### Backend (Azure Functions)

Convert FastAPI to Azure Functions using:
- `azure-functions`
- Function triggers for HTTP endpoints

## Performance Optimization

### 1. Use GPU Container Instances

```bash
az container create \
  --resource-group clinical-assist-rg \
  --name backend-gpu \
  --image clinicalassistacr.azurecr.io/backend:latest \
  --gpu-count 1 \
  --gpu-sku K80
```

### 2. Enable CDN for Frontend

```bash
az cdn profile create \
  --name clinical-assist-cdn \
  --resource-group clinical-assist-rg \
  --sku Standard_Microsoft

az cdn endpoint create \
  --name clinical-assist \
  --profile-name clinical-assist-cdn \
  --resource-group clinical-assist-rg \
  --origin frontend.azurecontainerapps.io
```

### 3. Configure Autoscaling

```bash
az containerapp update \
  --name clinical-assist-backend \
  --resource-group clinical-assist-rg \
  --min-replicas 2 \
  --max-replicas 10 \
  --scale-rule-name http-requests \
  --scale-rule-type http \
  --scale-rule-http-concurrency 50
```

## Monitoring & Logging

### 1. Enable Application Insights

```bash
az monitor app-insights component create \
  --app clinical-assist \
  --location eastus \
  --resource-group clinical-assist-rg
```

### 2. Configure Log Analytics

```bash
az containerapp logs show \
  --name clinical-assist-backend \
  --resource-group clinical-assist-rg \
  --follow
```

## Security Best Practices

1. **Enable HTTPS**: Use Azure Front Door or Application Gateway
2. **Environment Variables**: Store secrets in Azure Key Vault
3. **Network Security**: Configure VNet integration
4. **Authentication**: Add Azure AD authentication
5. **Rate Limiting**: Configure API Management

## Cost Optimization

- Use Azure Reserved Instances for predictable workloads
- Enable auto-shutdown for dev environments
- Use Azure Container Instances for burst workloads
- Monitor with Azure Cost Management

## Backup & Disaster Recovery

- Supabase handles database backups automatically
- Store model weights in Azure Blob Storage
- Configure geo-replication for high availability
- Implement health checks and auto-restart

## CI/CD Pipeline

### GitHub Actions Example

```yaml
name: Deploy to Azure

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Build Backend
        run: |
          cd backend
          docker build -t ${{ secrets.ACR_NAME }}.azurecr.io/backend:${{ github.sha }} .

      - name: Push to ACR
        run: |
          az acr login --name ${{ secrets.ACR_NAME }}
          docker push ${{ secrets.ACR_NAME }}.azurecr.io/backend:${{ github.sha }}

      - name: Update Container App
        run: |
          az containerapp update \
            --name clinical-assist-backend \
            --resource-group clinical-assist-rg \
            --image ${{ secrets.ACR_NAME }}.azurecr.io/backend:${{ github.sha }}
```

## Health Checks

Add to `app.py`:

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }
```

## Support & Maintenance

- Monitor error rates and response times
- Update dependencies regularly
- Retrain model with new data periodically
- Collect user feedback for improvements

## Troubleshooting

**Container fails to start**: Check logs with `az containerapp logs`
**High latency**: Enable GPU or add more replicas
**CORS errors**: Verify frontend URL in backend CORS settings
**Out of memory**: Increase memory allocation or optimize model
