import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision.models import resnet50
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print("Building Reference Database for Augmented ArcFace Model...")
    
    # Paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    checkpoint_path = base_dir / "classification" / "checkpoints" / "augmented_resnet50_arcface.pth"
    dataset_dir = base_dir / "datasets" / "migros_dataset_v6" / "dataset_arcface"
    output_db_path = base_dir / "classification" / "eval" / "outputs" / "reference_db_new.pt"
    
    if not dataset_dir.exists():
        print(f"Error: Unified dataset not found at {dataset_dir}")
        return
        
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint weights not found at {checkpoint_path}")
        print("Please copy your .pth file from Colab to classification/checkpoints/augmented_resnet50_arcface.pth")
        return
        
    # Load Basic Model
    backbone = resnet50(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, 512)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    backbone.load_state_dict(checkpoint, strict=True)
    backbone = backbone.to(DEVICE)
    backbone.eval()
    
    # Transforms
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Remove empty directories to avoid PyTorch ImageFolder strict empty-folder crashing
    for f in dataset_dir.iterdir():
        if f.is_dir() and not any(f.iterdir()):
            try: f.rmdir() 
            except Exception: pass
            
    # Dataloader
    dataset = datasets.ImageFolder(str(dataset_dir), transform=test_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Loaded {len(dataset)} items across {len(dataset.classes)} unique products.")
    
    all_embeddings = []
    all_labels = []
    all_class_names = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Feature Embeddings"):
            images = images.to(DEVICE)
            # Run inference
            embeddings = backbone(images)
            embeddings = F.normalize(embeddings, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.tolist())
            # Convert label indices back to their class string representation
            for lbl in labels:
                class_str = dataset.classes[lbl.item()]
                all_class_names.append(class_str)
                
    ref_embeddings = np.vstack(all_embeddings)
    
    # Save the exact same structure demo.py expects
    output_db_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'embeddings': ref_embeddings,
        'labels': all_labels,
        'class_names': all_class_names,
        'paths': ["NO_PATH_NEEDED"] * len(all_labels) # paths not strictly needed for webapp inference
    }, output_db_path)
    
    print(f"\nSuccess! Built high-dimensional Reference DB containing {len(ref_embeddings)} vectors.")
    print(f"Saved database to: {output_db_path}")
    print("Web Backend (demo.py) is now fully upgraded and ready!")

if __name__ == "__main__":
    main()
