"""
Enhanced ArcFace Training Script with Data Augmentation
Targeted for Google Colab GPU instances.
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import math

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.60):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * math.cos(self.m) - sine * math.sin(self.m)
        
        # Relax constraints
        phi = torch.where(cosine > 0, phi, cosine)
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

# --- COLLAB SETUP ---
# Update this DATA_DIR when running in Colab:
# e.g., DATA_DIR = "/content/dataset/migros_dataset_v6/crops"

def main():
    # Auto-detect if running on Google Colab vs Local PC
    if os.path.exists("/content/dataset_arcface"):
        DATA_DIR = "/content/dataset_arcface"
        print("🌍 Detected Google Colab Environment! Training on Cloud GPU...")
    else:
        DATA_DIR = "./datasets/migros_dataset_v6/dataset_arcface"
        print("💻 Detected Local PC Environment! Training on Local GPU...")
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. HEAVY DATA AUGMENTATION (Avoids Overfitting & Hallucination)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load Dataset
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Dataset not found at {DATA_DIR}. Make sure you extracted crops first.")
        return
        
    dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    num_classes = len(dataset.classes)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    print(f"Loaded {len(dataset)} images belonging to {num_classes} classes.")
    
    # Model (ResNet50 backbone + ArcFace Head)
    from torchvision.models import resnet50, ResNet50_Weights
    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Linear(backbone.fc.in_features, 512) # Feature extraction layer
    
    arcface_head = ArcFace(in_features=512, out_features=num_classes, m=0.60) # Increased margin!
    
    backbone = backbone.to(device)
    arcface_head = arcface_head.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': backbone.parameters(), 'lr': LEARNING_RATE * 0.1}, # Smaller LR for backbone
        {'params': arcface_head.parameters(), 'lr': LEARNING_RATE}
    ])
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        backbone.train()
        arcface_head.train()
        total_loss = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            features = backbone(images)
            outputs = arcface_head(features, labels)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        print(f"==== Epoch {epoch+1} Average Loss: {total_loss/len(dataloader):.4f} ====")
        
    # Save Model Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(backbone.state_dict(), "checkpoints/augmented_resnet50_arcface.pth")
    print("Training Complete! Model saved to checkpoints/augmented_resnet50_arcface.pth")

if __name__ == "__main__":
    main()
