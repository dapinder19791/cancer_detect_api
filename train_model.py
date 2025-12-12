import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import time

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.classes = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax",
            "Consolidation", "Edema", "Emphysema", "Fibrosis",
            "Pleural_Thickening", "Hernia"
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = self.image_dir / img_name
        image = Image.open(img_path).convert('RGB')

        labels = self.data.iloc[idx, 1:].values.astype('float32')

        if self.transform:
            image = self.transform(image)

        return image, torch.FloatTensor(labels)

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
        return self.densenet(x)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_outputs = np.vstack(all_outputs)

    try:
        auc_scores = []
        for i in range(all_labels.shape[1]):
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
                auc_scores.append(auc)

        mean_auc = np.mean(auc_scores) if auc_scores else 0.0
    except:
        mean_auc = 0.0

    return running_loss / len(dataloader), mean_auc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ChestXrayDataset(
        csv_file='data/train_labels.csv',
        image_dir='data/train_images',
        transform=transform_train
    )

    val_dataset = ChestXrayDataset(
        csv_file='data/val_labels.csv',
        image_dir='data/val_images',
        transform=transform_val
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    model = CheXNetModel(num_classes=14).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    num_epochs = 20
    best_auc = 0.0

    print("Starting training...")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val AUC: {val_auc:.4f}")
        print(f"  Time: {epoch_time:.2f}s")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'models/chexnet_best.pth')
            print(f"  → Saved best model (AUC: {best_auc:.4f})")

    print(f"\nTraining complete!")
    print(f"Best validation AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()
