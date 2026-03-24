"""
Planogram Generator

Takes detection + classification results (from demo.py) and generates
a planogram showing how products are arranged on each shelf.

Pipeline:
  1. Load CSV results (image_name, x1, y1, x2, y2, predicted_class, class_confidence)
  2. Detect shelf separator lines using vertical product density
  3. Assign each product to a shelf row based on its Y-center
  4. Sort products on each shelf left-to-right by x1
  5. Visualize as a planogram (shelf rows with proportionally sized product cells)

Usage (standalone):
    python planogram/planogram.py --csv demo_output/shelf_results.csv \
                                  --image path/to/shelf.jpg \
                                  --output planogram.png

    # Show actual product crop images inside cells:
    python planogram/planogram.py --csv demo_output/shelf_results.csv \
                                  --image path/to/shelf.jpg \
                                  --show-images

    # Process a folder of CSVs (one per image):
    python planogram/planogram.py --csv-folder demo_output/ \
                                  --image-folder images/ \
                                  --output-folder planograms/
"""

import os
import sys
import argparse
import hashlib
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# --- CONFIGURATION ---
SMOOTHING_FACTOR = 0.08   # KDE bandwidth (tightened to separate close wire shelves)
MIN_SHELF_SPACING = 30    # Minimum pixel gap between shelf separator lines
VALLEY_PROMINENCE = 3     # How prominent a valley must be to count as a shelf separator

PLANOGRAM_WIDTH_INCHES = 16   # Figure width
ROW_HEIGHT_INCHES = 2.2       # Height per shelf row (inches)
LABEL_FONT_SIZE = 6


# =============================================================================
# Color Utilities
# =============================================================================

def get_color_for_class(class_name: str) -> Tuple[float, float, float]:
    """Deterministic normalized RGB color (0-1) for a class name."""
    hash_val = int(hashlib.md5(class_name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(hash_val % (2**32))
    color = tuple(rng.integers(80, 230, size=3) / 255.0)
    return color


# =============================================================================
# Shelf Detection
# =============================================================================

def detect_shelf_lines(df: pd.DataFrame, img_h: int) -> np.ndarray:
    """
    Find Y-coordinates of gaps between shelf rows using vertical product density.

    The signal is the KDE of product center Y positions. Local minima in this
    signal are the gaps between shelves (the shelf boards/separators).

    Returns:
        Sorted array of Y-pixel positions (shelf separator lines).
        Empty array if fewer than 2 products or KDE fails.
    """
    if df.empty or len(df) < 2:
        return np.array([])

    df = df.copy()
    df['y_center'] = (df['y1'] + df['y2']) / 2
    y_positions = df['y_center'].values

    try:
        density = gaussian_kde(y_positions)
        density.covariance_factor = lambda: SMOOTHING_FACTOR
        density._compute_covariance()

        y_grid = np.linspace(0, img_h, img_h)
        signal = density(y_grid)
        signal = (signal / signal.max()) * 100   # Normalize to [0, 100]

        # Find valleys by inverting the signal and looking for peaks
        inverted = 100 - signal
        peaks, _ = find_peaks(inverted, distance=MIN_SHELF_SPACING,
                              prominence=VALLEY_PROMINENCE)

        return np.sort(y_grid[peaks])

    except Exception as e:
        print(f"  [WARNING] Shelf line detection failed: {e}")
        return np.array([])


# =============================================================================
# Shelf Assignment
# =============================================================================

def assign_shelves(df: pd.DataFrame, shelf_lines: np.ndarray) -> pd.DataFrame:
    """
    Assign each product to a shelf index.

    shelf_lines are the Y-pixel positions of the GAPS between shelves
    (not the product rows themselves). Products are assigned to the shelf
    whose gap they fall before.

    Shelf indices:
        0       → products with y_center < shelf_lines[0]
        1       → shelf_lines[0] <= y_center < shelf_lines[1]
        ...
        n       → y_center >= shelf_lines[-1]

    Returns:
        DataFrame with added columns: shelf (int), y_center (float).
        Sorted by (shelf, x1) for planogram order.
    """
    df = df.copy()
    df['y_center'] = (df['y1'] + df['y2']) / 2

    sorted_lines = np.sort(shelf_lines)

    def get_shelf_idx(y_center: float) -> int:
        for i, line_y in enumerate(sorted_lines):
            if y_center < line_y:
                return i
        return len(sorted_lines)  # Falls after all separator lines → last shelf

    df['shelf'] = df['y_center'].apply(get_shelf_idx)
    df = df.sort_values(['shelf', 'x1']).reset_index(drop=True)
    return df


# =============================================================================
# Planogram Visualization
# =============================================================================

def generate_planogram(
    df: pd.DataFrame,
    img_h: int,
    img_w: int,
    image_path: Optional[str] = None,
    output_path: Optional[str] = None,
    show_images: bool = False,
    title: str = "Planogram",
) -> plt.Figure:
    """
    Generate a planogram figure from classified detection results.

    Args:
        df:           DataFrame with columns [x1, y1, x2, y2,
                      predicted_class, class_confidence, ...]
        img_h:        Original image height in pixels.
        img_w:        Original image width in pixels.
        image_path:   Path to original shelf image (needed for show_images).
        output_path:  If set, saves the figure as a PNG/JPG.
        show_images:  If True and image_path is valid, renders actual product
                      crops inside planogram cells.
        title:        Figure title.

    Returns:
        matplotlib Figure object.
    """
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No products detected", ha='center', va='center',
                fontsize=14, color='gray')
        ax.axis('off')
        return fig

    # --- 1. Shelf detection & assignment ---
    shelf_lines = detect_shelf_lines(df, img_h)
    df_shelved = assign_shelves(df, shelf_lines)

    shelf_ids = sorted(df_shelved['shelf'].unique())
    num_shelves = len(shelf_ids)

    print(f"  Detected {len(shelf_lines)} shelf separator(s) → {num_shelves} shelf row(s)")

    # --- 2. Load original image for crop rendering ---
    orig_img = None
    if show_images and image_path and os.path.exists(image_path):
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            print(f"  [WARNING] Could not load image: {image_path}")

    # Orantılı görsel için x/y data birimlerinin fiziksel piksel karşılığı
    # x: 0..img_w data birimi = PLANOGRAM_WIDTH_INCHES inç
    # y: 0..1 data birimi (1 raf satırı) = ROW_HEIGHT_INCHES inç
    _DPI = 150
    _px_per_x = PLANOGRAM_WIDTH_INCHES * _DPI / max(img_w, 1)
    _px_per_y = ROW_HEIGHT_INCHES * _DPI

    # --- 3. Build color map ---
    class_colors: Dict[str, Tuple[float, float, float]] = {}
    for cls in df_shelved['predicted_class'].unique():
        class_colors[cls] = get_color_for_class(cls)

    # --- 4. Figure layout ---
    fig_height = num_shelves * ROW_HEIGHT_INCHES + 1.2
    fig, ax = plt.subplots(figsize=(PLANOGRAM_WIDTH_INCHES, fig_height))

    # Planogram coordinate system:
    #   X axis = pixel position along shelf (0 … img_w)
    #   Y axis = shelf row index (0 = top, num_shelves-1 = bottom), inverted
    ax.set_xlim(0, img_w)
    ax.set_ylim(num_shelves, 0)    # invert Y so shelf 0 is at top
    ax.set_aspect('auto')

    # --- 5. Draw each shelf row ---
    for row_idx, shelf_id in enumerate(shelf_ids):
        shelf_df = df_shelved[df_shelved['shelf'] == shelf_id].sort_values('x1')

        # Shelf band background (alternating shading)
        shade = 0.04 if row_idx % 2 == 0 else 0.08
        ax.axhspan(row_idx, row_idx + 1, alpha=shade, color='steelblue', zorder=0)

        # Shelf board line at bottom of row
        ax.axhline(row_idx + 1, color='#555555', linewidth=2.5, zorder=1)

        # Shelf label on left margin
        ax.text(
            -img_w * 0.015, row_idx + 0.5,
            f"Shelf {row_idx + 1}",
            va='center', ha='right',
            fontsize=9, fontweight='bold', color='#333333'
        )

        # --- 6. Draw products ---
        for _, product in shelf_df.iterrows():
            cls = product['predicted_class']
            conf = float(product.get('class_confidence', 0))
            x1, y1 = int(product['x1']), int(product['y1'])
            x2, y2 = int(product['x2']), int(product['y2'])

            bbox_w = max(x2 - x1, 1)
            color = class_colors[cls]

            # Cell occupies the shelf row proportionally to its pixel width
            cell_x = x1
            cell_w = bbox_w
            cell_y = row_idx + 0.04       # small top padding
            cell_h = 0.92                 # full row height minus padding

            # Render crop image inside cell (optional) — orantılı boyutlandırma
            if show_images and orig_img is not None:
                crop = orig_img[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    ih, iw = crop_rgb.shape[:2]
                    # Hücre boyutlarını fiziksel piksele çevir
                    max_w_px = cell_w * _px_per_x
                    max_h_px = cell_h * _px_per_y
                    # En-boy oranını koruyarak her iki boyuta da sığdır
                    scale = min(max_w_px / iw, max_h_px / ih)
                    disp_w = iw * scale / _px_per_x  # data birimine çevir
                    disp_h = ih * scale / _px_per_y
                    # Hücrede ortala
                    cx = cell_x + cell_w / 2
                    cy = cell_y + cell_h / 2
                    ax.imshow(
                        crop_rgb,
                        extent=[cx - disp_w/2, cx + disp_w/2,
                                cy + disp_h/2, cy - disp_h/2],
                        aspect='auto', zorder=2
                    )

            # Cell rectangle border
            face_color = (*color, 0.30) if not (show_images and orig_img is not None) else 'none'
            rect = FancyBboxPatch(
                (cell_x, cell_y), cell_w, cell_h,
                boxstyle="round,pad=0.005",
                linewidth=1.5,
                edgecolor=color,
                facecolor=face_color,
                zorder=3
            )
            ax.add_patch(rect)

            # Product label — sadece ürün adının son parçası (kategori yok)
            short_id = Path(cls).stem[:12]
            ax.text(
                cell_x + cell_w / 2,
                cell_y + cell_h * 0.88,
                short_id,
                ha='center', va='center',
                fontsize=LABEL_FONT_SIZE,
                fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.55, linewidth=0),
                zorder=5,
                clip_on=True
            )

    # --- 7. Axes formatting ---
    ax.set_xlabel("Horizontal Position on Shelf (pixels)", fontsize=10)
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.tick_params(axis='x', labelsize=8)

    # Left margin for shelf labels
    ax.set_xlim(-img_w * 0.06, img_w * 1.01)

    # --- 8. Legend ---
    legend_patches = [
        mpatches.Patch(facecolor=color, edgecolor='gray', label=cls)
        for cls, color in sorted(class_colors.items())
    ]
    if legend_patches:
        num_cols = max(1, min(len(legend_patches), 4))
        ax.legend(
            handles=legend_patches,
            loc='upper right',
            bbox_to_anchor=(1.0, 1.0),
            fontsize=7,
            ncol=num_cols,
            title="Products",
            title_fontsize=8,
            framealpha=0.9,
        )

    plt.tight_layout()

    # --- 9. Save ---
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Planogram saved: {output_path}")

    return fig


# =============================================================================
# Summary Statistics
# =============================================================================

def print_planogram_summary(df_shelved: pd.DataFrame) -> None:
    """Print a text summary of the planogram layout."""
    shelf_ids = sorted(df_shelved['shelf'].unique())
    total = len(df_shelved)
    print("\n" + "=" * 50)
    print("PLANOGRAM SUMMARY")
    print("=" * 50)
    print(f"Total products : {total}")
    print(f"Shelf rows     : {len(shelf_ids)}")
    print()

    for shelf_id in shelf_ids:
        shelf_df = df_shelved[df_shelved['shelf'] == shelf_id].sort_values('x1')
        print(f"  Shelf {shelf_id + 1} ({len(shelf_df)} products):")
        for _, p in shelf_df.iterrows():
            print(f"    [{p['x1']:4d}-{p['x2']:4d}]  {p['predicted_class']}  "
                  f"(conf={p.get('class_confidence', 0):.2f})")
    print("=" * 50)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate planogram from detection+classification CSV results."
    )

    # Single image mode
    single = parser.add_argument_group("Single image mode")
    single.add_argument('--csv', type=str,
                        help='Path to results CSV (from demo.py)')
    single.add_argument('--image', type=str,
                        help='Path to original shelf image')
    single.add_argument('--output', type=str, default=None,
                        help='Output path for planogram image (PNG/JPG)')

    # Folder mode
    folder = parser.add_argument_group("Folder mode")
    folder.add_argument('--csv-folder', type=str,
                        help='Folder containing per-image result CSVs')
    folder.add_argument('--image-folder', type=str,
                        help='Folder containing original shelf images')
    folder.add_argument('--output-folder', type=str,
                        help='Output folder for planogram images')

    # Shared options
    parser.add_argument('--show-images', action='store_true',
                        help='Render actual product crop images inside planogram cells')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not show interactive plot window')

    args = parser.parse_args()

    # --- Folder mode ---
    if args.csv_folder:
        csv_folder = Path(args.csv_folder)
        image_folder = Path(args.image_folder) if args.image_folder else None
        out_folder = Path(args.output_folder) if args.output_folder else csv_folder / 'planograms'
        out_folder.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(csv_folder.glob('*_results.csv'))
        if not csv_files:
            # Fall back: any CSV
            csv_files = sorted(csv_folder.glob('*.csv'))

        if not csv_files:
            print(f"[ERROR] No CSV files found in {csv_folder}")
            return

        print(f"Found {len(csv_files)} CSV file(s)")

        for csv_path in csv_files:
            stem = csv_path.stem.replace('_results', '')
            print(f"\nProcessing {csv_path.name}...")

            df = pd.read_csv(csv_path)
            if df.empty:
                print("  Empty CSV, skipping.")
                continue

            # Try to resolve original image
            img_path = None
            if image_folder:
                for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
                    candidate = image_folder / (stem + ext)
                    if candidate.exists():
                        img_path = str(candidate)
                        break
                if img_path is None:
                    # Try using image_name column if available
                    if 'image_name' in df.columns:
                        candidate = image_folder / df.iloc[0]['image_name']
                        if candidate.exists():
                            img_path = str(candidate)

            # Get image dimensions
            img_h = img_w = None
            if img_path:
                im = cv2.imread(img_path)
                if im is not None:
                    img_h, img_w = im.shape[:2]
            if img_h is None:
                # Fall back: compute from bbox extents
                img_h = int(df['y2'].max()) + 50
                img_w = int(df['x2'].max()) + 50

            out_path = str(out_folder / f"{stem}_planogram.png")
            fig = generate_planogram(
                df, img_h, img_w,
                image_path=img_path,
                output_path=out_path,
                show_images=args.show_images,
                title=f"Planogram — {stem}",
            )
            plt.close(fig)

        print(f"\nDone! Planograms saved to: {out_folder}")
        return

    # --- Single image mode ---
    if not args.csv:
        parser.print_help()
        return

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[ERROR] CSV is empty.")
        return

    img_path = args.image
    img_h = img_w = None

    if img_path and os.path.exists(img_path):
        im = cv2.imread(img_path)
        if im is not None:
            img_h, img_w = im.shape[:2]

    if img_h is None:
        img_h = int(df['y2'].max()) + 50
        img_w = int(df['x2'].max()) + 50

    stem = csv_path.stem.replace('_results', '')
    out_path = args.output or str(csv_path.parent / f"{stem}_planogram.png")

    print(f"Generating planogram for {csv_path.name}...")
    print(f"  Image size: {img_w}x{img_h} pixels")
    print(f"  Products:   {len(df)}")

    fig = generate_planogram(
        df, img_h, img_w,
        image_path=img_path,
        output_path=out_path,
        show_images=args.show_images,
        title=f"Planogram — {stem}",
    )

    # Print text summary
    shelf_lines = detect_shelf_lines(df, img_h)
    df_shelved = assign_shelves(df, shelf_lines)
    print_planogram_summary(df_shelved)

    if not args.no_display:
        plt.show()

    plt.close(fig)


if __name__ == '__main__':
    main()
