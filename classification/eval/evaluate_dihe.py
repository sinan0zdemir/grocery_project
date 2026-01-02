"""
DIHE Evaluation Script

Evaluates a trained DIHE model on the Grocery_products dataset.
Uses the MACVGG encoder from the cvpce library.

Usage:
    python evaluate_dihe.py [--encoder-weights PATH] [--batch-size N]

Expects:
    - datasets/Grocery_products/Training
    - datasets/Grocery_products/Testing
    - datasets/Grocery_products/Annotations
    - classification/checkpoints/ (encoder weights as .tar file)
    - cvpce library in the project root
"""

import sys
import os
import types
import argparse
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CVPCE_ROOT = PROJECT_ROOT.parent / "cvpce"

# Add cvpce to path
if str(CVPCE_ROOT) not in sys.path:
    sys.path.insert(0, str(CVPCE_ROOT))

# =============================================================================
# Compatibility Patches (Torchvision 0.9+ compatibility)
# =============================================================================
import torch
import torchvision.models

# Patch for torchvision.models.utils (removed in newer versions)
fake_utils = types.ModuleType("torchvision.models.utils")
if hasattr(torch.hub, 'load_state_dict_from_url'):
    fake_utils.load_state_dict_from_url = torch.hub.load_state_dict_from_url
else:
    from torch.hub import load_state_dict_from_url
    fake_utils.load_state_dict_from_url = load_state_dict_from_url

sys.modules["torchvision.models.utils"] = fake_utils
torchvision.models.utils = fake_utils

# Patch for vgg.model_urls (removed in newer versions)
import torchvision.models.vgg as vgg
if not hasattr(vgg, 'model_urls'):
    vgg.model_urls = {
        'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
        'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
        'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
        'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
        'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    }

print("✅ Compatibility patches applied.")

# =============================================================================
# CVPCE Imports
# =============================================================================
try:
    from cvpce import datautils
    from cvpce.models import classification
    print("✅ CVPCE modules loaded.")
except ImportError as e:
    print(f"❌ Error loading CVPCE modules: {e}")
    print(f"   Make sure cvpce is at: {CVPCE_ROOT}")
    sys.exit(1)

from torch.utils.data import DataLoader
from torchvision import ops as tvops

# =============================================================================
# Dataset Paths
# =============================================================================
DATASET_DIR = PROJECT_ROOT / "datasets" / "Grocery_products"
CHECKPOINT_DIR = SCRIPT_DIR.parent / "checkpoints"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

TRAINING_DIR = DATASET_DIR / "Training"
TESTING_DIR = DATASET_DIR / "Testing"
ANNOTATIONS_DIR = DATASET_DIR / "Annotations"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Filename Fix (Required for cvpce compatibility)
# =============================================================================
def fix_test_image_filenames(testing_dir: Path):
    """
    Fix test image filenames to match cvpce expected format.
    
    The cvpce library expects: store2_18.jpg
    But original files might be: 18.jpg
    
    This function renames files only if needed.
    """
    print("\n🔧 Checking test image filenames...")
    
    for store_folder in testing_dir.iterdir():
        if not store_folder.is_dir() or 'store' not in store_folder.name:
            continue
        
        images_dir = store_folder / 'images'
        if not images_dir.exists():
            continue
        
        count = 0
        for img_file in images_dir.iterdir():
            if not img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                continue
            
            # Check if filename already has the store prefix
            if not img_file.name.startswith(store_folder.name):
                new_name = f"{store_folder.name}_{img_file.name}"
                new_path = images_dir / new_name
                
                img_file.rename(new_path)
                count += 1
        
        if count > 0:
            print(f"   ✅ {store_folder.name}: Renamed {count} images (e.g., '18.jpg' → '{store_folder.name}_18.jpg')")
        else:
            print(f"   ℹ️ {store_folder.name}: Files already correct")
    
    print("🎉 Filename check complete.")


# =============================================================================
# Custom Classifier (CPU/CUDA compatible)
# =============================================================================
class Classifier:
    """Custom classifier that properly handles CPU/CUDA device selection."""
    
    def __init__(self, encoder, sample_set, device, batch_size=32, num_workers=4, k=1):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.k = k
        self.encoder = encoder
        
        self.embedding, self.annotations = self.build_index(sample_set)
    
    def build_index(self, sample_set):
        """Build embedding index from sample set."""
        loader = DataLoader(
            sample_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=datautils.gp_annotated_collate_fn,
            pin_memory=self.device.type == 'cuda'
        )
        
        embeddings_list = []
        annotations = []
        
        with torch.no_grad():
            for i, (imgs, _, _, anns) in enumerate(loader):
                if i % 100 == 0:
                    print(f"  Building index: batch {i}...")
                
                imgs = imgs.to(device=self.device)
                emb = self.encoder(imgs).detach().cpu()  # Keep on CPU to save memory
                embeddings_list.append(emb)
                annotations += anns
        
        embedding = torch.cat(embeddings_list, dim=0).to(self.device)
        return embedding, annotations
    
    def classify(self, images):
        """Classify images using k-nearest neighbors."""
        results = []
        
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i + self.batch_size].to(device=self.device)
                # Scale to tanh range (-1 to 1)
                batch = batch * 2 - 1
                emb = self.encoder(batch).detach()
                
                # Compute nearest neighbors using cosine similarity
                distances = 1 - torch.nn.functional.cosine_similarity(
                    self.embedding.unsqueeze(0),  # [1, N, D]
                    emb.unsqueeze(1),              # [B, 1, D]
                    dim=-1
                )
                
                nearest_indices = distances.argsort(dim=-1)[:, :self.k]
                results += [[self.annotations[j.item()] for j in n] for n in nearest_indices]
        
        return results


def eval_dihe_custom(encoder, sampleset, testset, device, batch_size, num_workers, k=(1,), verbose=True):
    """
    Custom DIHE evaluation function with proper device handling.
    """
    if verbose:
        print('Preparing classifier...')
    
    encoder.requires_grad_(False)
    
    classifier = Classifier(
        encoder, sampleset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        k=max(k)
    )
    
    total = 0
    correct = {knn: 0 for knn in k}
    missed = {}
    misclassification = {}
    total_per_ann = {}
    
    if verbose:
        print('Eval start!')
    
    for i, (img, target_anns, boxes) in enumerate(testset):
        if verbose and i % 10 == 0:
            print(f'{i}...')
        
        # Clip boxes to image dimensions
        boxes = tvops.clip_boxes_to_image(boxes, (img.shape[1], img.shape[2]))
        
        # Crop and resize each box
        imgs = torch.stack([
            datautils.resize_for_classification(img[:, y1:y2, x1:x2])
            for x1, y1, x2, y2 in boxes
        ])
        
        pred_anns = classifier.classify(imgs)
        
        total += len(target_anns)
        for a1, a2 in zip(target_anns, pred_anns):
            if a1 not in total_per_ann:
                total_per_ann[a1] = 0
            total_per_ann[a1] += 1
            
            for knn in k:
                if a1 in a2[:knn]:
                    correct[knn] += 1
            
            if a1 != a2[0]:
                if a1 not in missed:
                    missed[a1] = 0
                    misclassification[a1] = {}
                if a2[0] not in misclassification[a1]:
                    misclassification[a1][a2[0]] = 0
                missed[a1] += 1
                misclassification[a1][a2[0]] += 1
    
    del classifier
    encoder.requires_grad_(True)
    
    accuracy = {knn: c / total for knn, c in correct.items()}
    
    if verbose:
        print(f'Total annotations: {total}, Correctly guessed: {correct}, Accuracy: {accuracy}')
        if missed:
            most_missed = sorted(
                ((v / total_per_ann[k], v, k) for k, v in missed.items()),
                reverse=True
            )[:10]
            print(f'Most missed: {", ".join(f"{a} ({n}, {p * 100:.1f}%)" for p, n, a in most_missed)}')
            for _, n, k_class in most_missed[:3]:
                common_misclassifications = sorted(
                    ((v / n, v, k) for k, v in misclassification[k_class].items()),
                    reverse=True
                )[:3]
                print(f'{k_class}: Commonly mistaken for {", ".join(f"{a} ({n}, {p * 100:.1f}%)" for p, n, a in common_misclassifications)}')
    
    return accuracy


def find_encoder_weights(checkpoint_dir: Path) -> Path:
    """Find encoder weights file (supports .tar format)."""
    # Look for .tar files first (DIHE format)
    tar_files = list(checkpoint_dir.glob("*.tar"))
    if tar_files:
        # Prefer epoch files first, then checkpoint
        epoch_files = [f for f in tar_files if 'epoch' in f.name.lower()]
        if epoch_files:
            return sorted(epoch_files)[-1]  # Latest epoch
        return tar_files[0]
    
    # Fallback to .pth files
    pth_files = list(checkpoint_dir.glob("*.pth"))
    if pth_files:
        return pth_files[0]
    
    return None


def load_encoder(weights_path: Path, model_type: str = 'vgg16', batch_norm: bool = False):
    """Load the DIHE encoder with trained weights."""
    print(f"\n� Loading encoder from: {weights_path}")
    
    # Create the encoder
    if model_type == 'vgg16':
        encoder = classification.macvgg_embedder(
            model='vgg16_bn' if batch_norm else 'vgg16',
            pretrained=True
        )
    elif model_type == 'resnet50':
        encoder = classification.macresnet_encoder(pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    
    state = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(state, dict):
        if 'model_state_dict' in state:
            encoder.load_state_dict(state['model_state_dict'])
        elif 'encoder_state_dict' in state:
            encoder.load_state_dict(state['encoder_state_dict'])
        else:
            # Try loading directly (might be just the state dict)
            encoder.load_state_dict(state)
    else:
        encoder.load_state_dict(state)
    
    encoder = encoder.to(DEVICE)
    encoder.eval()
    
    print(f"✅ Encoder loaded successfully!")
    print(f"   Model type: {model_type}")
    print(f"   Batch norm: {batch_norm}")
    print(f"   Embedding size: {encoder.embedding_size}")
    
    return encoder


def main():
    parser = argparse.ArgumentParser(description="DIHE Model Evaluation")
    parser.add_argument("--encoder-weights", type=str, default=None,
                        help="Path to encoder weights (.tar file)")
    parser.add_argument("--model-type", type=str, default="vgg16",
                        choices=["vgg16", "resnet50"],
                        help="Model architecture type")
    parser.add_argument("--batch-norm", action="store_true",
                        help="Use batch normalization (for vgg16_bn)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--k", type=str, default="1,5,10",
                        help="Top-k values for accuracy (comma-separated)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("DIHE Evaluation Script")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # ==========================================================================
    # Step 1: Validate paths
    # ==========================================================================
    print("\n📁 Checking dataset paths...")
    
    if not TRAINING_DIR.exists():
        print(f"❌ Training directory not found: {TRAINING_DIR}")
        return
    if not TESTING_DIR.exists():
        print(f"❌ Testing directory not found: {TESTING_DIR}")
        return
    if not ANNOTATIONS_DIR.exists():
        print(f"❌ Annotations directory not found: {ANNOTATIONS_DIR}")
        return
    
    print(f"✅ Training: {TRAINING_DIR}")
    print(f"✅ Testing: {TESTING_DIR}")
    print(f"✅ Annotations: {ANNOTATIONS_DIR}")
    
    # ==========================================================================
    # Step 2: Find and load encoder weights
    # ==========================================================================
    if args.encoder_weights:
        weights_path = Path(args.encoder_weights)
    else:
        weights_path = find_encoder_weights(CHECKPOINT_DIR)
    
    if weights_path is None:
        print(f"❌ No encoder weights found in: {CHECKPOINT_DIR}")
        print("   Expected .tar files (e.g., epoch_9.tar, checkpoint.tar)")
        return
    
    try:
        encoder = load_encoder(weights_path, args.model_type, args.batch_norm)
    except Exception as e:
        print(f"❌ Failed to load encoder: {e}")
        return
    
    # ==========================================================================
    # Step 3: Prepare datasets
    # ==========================================================================
    # Fix test image filenames if needed (cvpce expects store2_18.jpg format)
    fix_test_image_filenames(TESTING_DIR)
    
    print("\n📊 Preparing datasets...")
    
    # Training set (reference database)
    sampleset = datautils.GroceryProductsDataset(
        [str(TRAINING_DIR)],
        include_annotations=True
    )
    print(f"✅ Sample set loaded: {len(sampleset)} images")
    
    # Test set
    testset = datautils.GroceryProductsTestSet(
        str(TESTING_DIR),
        str(ANNOTATIONS_DIR)
    )
    print(f"✅ Test set loaded: {len(testset)} test images")
    
    # ==========================================================================
    # Step 4: Run evaluation
    # ==========================================================================
    print("\n🔍 Running Evaluation...")
    
    # Parse k values
    k_values = tuple(int(k.strip()) for k in args.k.split(","))
    print(f"   Top-k values: {k_values}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Num workers: {args.num_workers}")
    
    accuracy = eval_dihe_custom(
        encoder,
        sampleset,
        testset,
        device=DEVICE,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        k=k_values
    )
    
    # ==========================================================================
    # Step 5: Display and save results
    # ==========================================================================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for k_val, acc in accuracy.items():
        print(f"  Top-{k_val} Accuracy: {acc * 100:.2f}%")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = OUTPUT_DIR / "dihe_results.txt"
    
    with open(results_file, "w") as f:
        f.write("DIHE Evaluation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Encoder weights: {weights_path}\n")
        f.write(f"Model type: {args.model_type}\n")
        f.write(f"Batch norm: {args.batch_norm}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write("\nAccuracy:\n")
        for k_val, acc in accuracy.items():
            f.write(f"  Top-{k_val}: {acc:.4f} ({acc * 100:.2f}%)\n")
    
    print(f"\n� Results saved to: {results_file}")
    print("✅ Evaluation complete!")


if __name__ == "__main__":
    main()
