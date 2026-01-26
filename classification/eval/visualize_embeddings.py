"""
t-SNE Embedding Visualization for HAL (Hierarchical Auxiliary Learning)

This script generates a t-SNE visualization showing clusters of products
grouped by category, demonstrating the semantic organization provided by HAL.

Visualizes reference_db.pt from eval/outputs directory.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches


# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
REFERENCE_DB_PATH = OUTPUT_DIR / "reference_db.pt"

# t-SNE parameters
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_RANDOM_STATE = 42

# Visualization parameters
FIGURE_SIZE = (10, 8)
POINT_SIZE = 50
ALPHA = 1.0


# =============================================================================
# Helper Functions
# =============================================================================
def extract_category(class_name: str) -> str:
    """
    Extract leaf category (parent folder of the image file) from class name.
    Example: 'Food/Rice/product.jpg' -> 'Rice'
    """
    parts = class_name.replace('\\', '/').split('/')
    # Get the parent folder of the image file (leaf category)
    if len(parts) >= 2:
        return parts[-2]  # Parent folder of the file
    return 'Unknown'


def get_color_palette(num_colors: int) -> List[str]:
    """Generate a visually distinct color palette."""
    # Use a perceptually uniform colormap
    cmap = plt.cm.get_cmap('tab20', min(20, num_colors))
    colors = [cmap(i) for i in range(min(20, num_colors))]
    
    # If more than 20 categories, extend with another colormap
    if num_colors > 20:
        cmap2 = plt.cm.get_cmap('Set3', 12)
        colors.extend([cmap2(i) for i in range(min(12, num_colors - 20))])
    
    if num_colors > 32:
        cmap3 = plt.cm.get_cmap('Pastel1', 9)
        colors.extend([cmap3(i) for i in range(num_colors - 32)])
    
    return colors[:num_colors]


def load_reference_db(path: Path) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load reference database containing embeddings and metadata."""
    print(f"Loading reference database from {path}...")
    
    if not path.exists():
        raise FileNotFoundError(f"Reference database not found: {path}")
    
    data = torch.load(path, map_location='cpu', weights_only=False)
    
    embeddings = data['embeddings']
    class_names = data['class_names']
    paths = data.get('paths', [])
    
    # Convert to numpy if tensor
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()
    
    print(f"  Loaded {len(embeddings)} embeddings")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    
    return embeddings, class_names, paths


def compute_tsne(embeddings: np.ndarray, perplexity: int = TSNE_PERPLEXITY, 
                 n_iter: int = TSNE_N_ITER, random_state: int = TSNE_RANDOM_STATE) -> np.ndarray:
    """Compute t-SNE dimensionality reduction."""
    print(f"Computing t-SNE (perplexity={perplexity}, n_iter={n_iter})...")
    
    # Adjust perplexity if necessary
    n_samples = embeddings.shape[0]
    actual_perplexity = min(perplexity, (n_samples - 1) // 3)
    if actual_perplexity != perplexity:
        print(f"  Adjusted perplexity to {actual_perplexity} for {n_samples} samples")
    
    tsne = TSNE(
        n_components=2,
        perplexity=actual_perplexity,
        max_iter=n_iter,
        random_state=random_state,
        learning_rate='auto',
        init='pca'
    )
    
    embeddings_2d = tsne.fit_transform(embeddings)
    print(f"  t-SNE complete. Output shape: {embeddings_2d.shape}")
    
    return embeddings_2d


def create_visualization(
    embeddings_2d: np.ndarray,
    categories: List[str],
    category_to_idx: Dict[str, int],
    output_path: Path,
    title: str = "t-SNE Visualization of Product Embeddings by Category"
) -> None:
    """Create and save t-SNE visualization."""
    print(f"Creating visualization...")
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, facecolor='white')
    
    # Get unique categories and colors
    unique_categories = sorted(category_to_idx.keys())
    num_categories = len(unique_categories)
    colors = get_color_palette(num_categories)
    category_colors = {cat: colors[i] for i, cat in enumerate(unique_categories)}
    
    # Create color array for all points
    point_colors = [category_colors[cat] for cat in categories]
    
    # Scatter plot
    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=point_colors,
        s=POINT_SIZE,
        alpha=ALPHA,
        edgecolors='white',
        linewidths=0.3
    )
    
    # Create legend
    legend_handles = [
        mpatches.Patch(color=category_colors[cat], label=f"{cat} ({sum(1 for c in categories if c == cat)})")
        for cat in unique_categories
    ]
    
    # Position legend outside plot
    ax.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        title='Categories',
        title_fontsize=10,
        framealpha=0.9
    )
    
    # Styling
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    
    # Remove spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)
    
    ax.tick_params(axis='both', which='both', length=0, labelsize=9)
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Add annotation
    textstr = f'Total samples: {len(categories)}\nCategories: {num_categories}'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save as PNG and SVG
    png_path = output_path.with_suffix('.png')
    svg_path = output_path.with_suffix('.svg')
    
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {png_path}")
    print(f"  Saved: {svg_path}")


def print_category_stats(categories: List[str], category_to_idx: Dict[str, int]) -> None:
    """Print statistics about category distribution."""
    print("\nCategory Distribution:")
    print("-" * 40)
    
    category_counts = {}
    for cat in categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for cat in sorted(category_counts.keys()):
        print(f"  {cat}: {category_counts[cat]} samples")
    
    print("-" * 40)
    print(f"  Total: {len(categories)} samples in {len(category_counts)} categories")


def main():
    """Main function to generate t-SNE visualization."""
    print("=" * 60)
    print("t-SNE Embedding Visualization for HAL")
    print("=" * 60)
    
    # Load reference database
    embeddings, class_names, paths = load_reference_db(REFERENCE_DB_PATH)
    
    # Extract categories from class names
    categories = [extract_category(cn) for cn in class_names]
    
    # Build category mapping
    unique_categories = sorted(set(categories))
    category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
    
    print_category_stats(categories, category_to_idx)
    
    # Compute t-SNE
    embeddings_2d = compute_tsne(embeddings)
    
    # Create visualization
    output_path = OUTPUT_DIR / "tsne_by_category"
    create_visualization(
        embeddings_2d,
        categories,
        category_to_idx,
        output_path,
        title="t-SNE: Product Embeddings Clustered by Category (HAL)"
    )
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
