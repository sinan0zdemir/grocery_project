"""
Quick planogram demo using the hand-labelled _annotations.csv from Roboflow.

Usage:
    uv run python demo_planogram.py --csv path/to/_annotations.csv
    uv run python demo_planogram.py --csv path/to/_annotations.csv --image-folder path/to/images/
"""

import argparse
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from planogram import generate_planogram, detect_shelf_lines, assign_shelves, print_planogram_summary


def convert_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Roboflow TF-OD annotation format to planogram format.

    Input columns:  filename, width, height, class, xmin, ymin, xmax, ymax
    Output columns: x1, y1, x2, y2, predicted_class, class_confidence, image_name, img_w, img_h
    """
    out = pd.DataFrame()
    out['x1']              = df['xmin']
    out['y1']              = df['ymin']
    out['x2']              = df['xmax']
    out['y2']              = df['ymax']
    out['predicted_class'] = df['class'].astype(str)
    out['class_confidence'] = 1.0
    out['image_name']      = df['filename']
    out['img_w']           = df['width']
    out['img_h']           = df['height']
    return out


def main():
    parser = argparse.ArgumentParser(description="Planogram demo from annotated CSV")
    parser.add_argument('--csv', required=True, help='Path to _annotations.csv')
    parser.add_argument('--image-folder', default=None,
                        help='Folder containing the original images (optional, for --show-images)')
    parser.add_argument('--show-images', action='store_true',
                        help='Render actual product crops inside planogram cells')
    parser.add_argument('--output-folder', default='planogram_output',
                        help='Where to save the planogram PNGs (default: planogram_output/)')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    raw = pd.read_csv(csv_path)
    print(f"Loaded {len(raw)} annotations from {csv_path.name}")
    print(f"Images in CSV: {raw['filename'].nunique()}")

    df = convert_annotations(raw)

    out_folder = Path(args.output_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    for filename, group in df.groupby('image_name'):
        stem = Path(filename).stem
        img_h = int(group['img_h'].iloc[0])
        img_w = int(group['img_w'].iloc[0])

        # Try to find the image file
        img_path = None
        if args.image_folder:
            for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
                candidate = Path(args.image_folder) / filename
                if candidate.exists():
                    img_path = str(candidate)
                    break
                candidate = Path(args.image_folder) / (Path(filename).stem + ext)
                if candidate.exists():
                    img_path = str(candidate)
                    break

        out_path = str(out_folder / f"{stem}_planogram.png")
        print(f"\nProcessing: {filename}  ({img_w}x{img_h}, {len(group)} products)")

        fig, _ = generate_planogram(
            group[['x1', 'y1', 'x2', 'y2', 'predicted_class', 'class_confidence']],
            img_h, img_w,
            image_path=img_path,
            output_path=out_path,
            show_images=args.show_images and img_path is not None,
            title=f"Planogram — {Path(filename).name}",
        )
        plt.close(fig)

        shelf_lines = detect_shelf_lines(
            group[['x1', 'y1', 'x2', 'y2', 'predicted_class', 'class_confidence']],
            img_h
        )
        df_shelved = assign_shelves(
            group[['x1', 'y1', 'x2', 'y2', 'predicted_class', 'class_confidence']],
            shelf_lines
        )
        print_planogram_summary(df_shelved)

    print(f"\nDone! Planograms saved to: {out_folder}/")


if __name__ == '__main__':
    main()
