"""
Embedding Visualization Script

Visualizes the reference database embeddings using UMAP dimensionality reduction.
Creates an interactive plot with category-based coloring.

Usage:
    python visualize_embeddings.py [--db-path PATH] [--output PATH]
"""

import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
import umap

# Configuration
SCRIPT_DIR = Path(__file__).parent
DEFAULT_DB_PATH = SCRIPT_DIR / "outputs" / "reference_db.pt"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "embeddings_visualization.png"


def extract_category(class_name: str, level: int = -2) -> str:
    """
    Extract category from class path.
    
    Args:
        class_name: Full path like 'Food/Rice/12.jpg'
        level: Which level to extract:
            0 = top-level (e.g., 'Food')
            -2 = leaf category (e.g., 'Rice')
            -1 = product name (e.g., '12')
    """
    parts = class_name.replace('\\', '/').split('/')
    # Remove file extension from last part
    if parts and '.' in parts[-1]:
        parts[-1] = parts[-1].rsplit('.', 1)[0]
    
    if len(parts) >= abs(level):
        return parts[level]
    return parts[0] if parts else 'Unknown'


def load_reference_db(db_path: Path) -> dict:
    """Load the reference database."""
    print(f"📂 Loading reference database from: {db_path}")
    data = torch.load(db_path, map_location='cpu', weights_only=False)
    
    embeddings = data['embeddings']
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()
    
    class_names = data['class_names']
    
    print(f"   ✅ Loaded {len(embeddings)} embeddings with {embeddings.shape[1]} dimensions")
    return {
        'embeddings': embeddings,
        'class_names': class_names,
        'labels': data.get('labels', list(range(len(class_names)))),
        'paths': data.get('paths', [])
    }


def visualize_embeddings(
    embeddings: np.ndarray,
    class_names: list,
    output_path: Path,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
):
    """Create UMAP visualization with category coloring."""
    
    # Extract categories
    categories = [extract_category(name) for name in class_names]
    unique_categories = sorted(set(categories))
    category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
    category_indices = [category_to_idx[cat] for cat in categories]
    
    print(f"\n📊 Found {len(unique_categories)} categories:")
    category_counts = defaultdict(int)
    for cat in categories:
        category_counts[cat] += 1
    for cat in sorted(category_counts.keys()):
        print(f"   • {cat}: {category_counts[cat]} samples")
    
    # Compute UMAP
    print(f"\n🔄 Computing UMAP projection (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=True
    )
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create figure
    print("\n🎨 Creating visualization...")
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use a colormap with enough colors
    cmap = plt.cm.get_cmap('tab20' if len(unique_categories) <= 20 else 'nipy_spectral')
    
    # Plot each category
    for idx, category in enumerate(unique_categories):
        mask = np.array(categories) == category
        color = cmap(idx / len(unique_categories))
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=f"{category} ({mask.sum()})",
            s=15,
            alpha=0.7
        )
    
    # Styling
    ax.set_title("Reference Database Embeddings (UMAP Projection)", fontsize=14, fontweight='bold')
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    
    # Legend (outside plot if many categories)
    if len(unique_categories) > 10:
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            fontsize=8,
            markerscale=1.5
        )
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        ax.legend(loc='best', fontsize=9, markerscale=1.5)
        plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), bbox_inches='tight')  # Also save as SVG
    print(f"\n✅ Saved visualization to: {output_path}")
    print(f"✅ Saved SVG to: {output_path.with_suffix('.svg')}")
    
    plt.show()
    
    return embeddings_2d


def main():
    parser = argparse.ArgumentParser(description="Visualize embedding database")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH),
                        help="Path to reference_db.pt file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output image path")
    parser.add_argument("--n-neighbors", type=int, default=15,
                        help="UMAP n_neighbors parameter")
    parser.add_argument("--min-dist", type=float, default=0.1,
                        help="UMAP min_dist parameter")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Embedding Visualization")
    print("=" * 60)
    
    # Load data
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    data = load_reference_db(db_path)
    
    # Visualize
    visualize_embeddings(
        data['embeddings'],
        data['class_names'],
        Path(args.output),
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist
    )
    
    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
