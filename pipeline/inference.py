import sys
from pathlib import Path
import pandas as pd

# Add root project dir to path to import demo and planogram modules
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from demo import load_classification_model, run_detection, process_image, YOLO, CLASSIFICATION_CHECKPOINT, REFERENCE_DB_PATH, DETECTION_WEIGHTS
from planogram import detect_shelf_lines, assign_shelves, generate_planogram
from comparator import compare_shelves, load_schema, evaluate_hybrid_shelves, generate_schema_from_df
import pandas as pd

import urllib.request
import os

# Global singletons for models
_detection_model = None
_classification_model = None
_ref_embeddings = None
_ref_class_names = None
_idx_to_class = None

def initialize_models():
    """Load models into memory once at startup."""
    global _detection_model, _classification_model, _ref_embeddings, _ref_class_names, _idx_to_class
    
    if _detection_model is None:
        print("Initializing YOLO Model...")
        if DETECTION_WEIGHTS.exists():
            _detection_model = YOLO(str(DETECTION_WEIGHTS))
        else:
            print(f"ERROR: detection weights not found at {DETECTION_WEIGHTS}")
            
    if _classification_model is None:
        print("Initializing ArcFace Model...")
        # Workaround for demo.py path cache/evaluation issues:
        actual_ref_db = ROOT_DIR / "classification" / "eval" / "outputs" / "reference_db_new.pt"
        if actual_ref_db.exists() and CLASSIFICATION_CHECKPOINT.exists():
            _classification_model, _ref_embeddings, _ref_class_names, _idx_to_class = \
                load_classification_model(CLASSIFICATION_CHECKPOINT, actual_ref_db)
        else:
            print(f"ERROR: Arcface weights or reference DB not found. DB={actual_ref_db.exists()}")

def run_analysis(image_path: str, schemas_dir: str, output_folder: Path) -> dict:
    """Run full pipeline: Detection -> Classification -> Planogram Compare"""
    
    schemas = []
    schemas_path = Path(schemas_dir)
    if schemas_path.exists() and schemas_path.is_dir():
        for file in schemas_path.glob("*.json"):
            schemas.append(load_schema(str(file)))
            
    if not schemas:
        schemas = [{"name": "Default", "rows": []}]
    
    # 1. Ensure models
    initialize_models()
    
    # Create subfolders for outputs
    cls_folder = output_folder / "classification"
    det_folder = output_folder / "detection"
    plan_folder = output_folder / "planogram"
    for folder in [cls_folder, det_folder, plan_folder]:
        folder.mkdir(parents=True, exist_ok=True)
        
    # 2. Run Image processing
    df, timing = process_image(
        image_path,
        _detection_model,
        _classification_model,
        _ref_embeddings,
        _ref_class_names,
        cls_folder,
        det_folder,
        plan_folder
    )
    
    if df.empty:
        return {"status": "error", "message": "No products detected."}
        
    csv_path = ROOT_DIR / "datasets" / "migros_dataset_v6" / "Annotations" / "SDP_Product&ID_Dataset_fix.csv"
    global_mapping = {}
    if csv_path.exists():
        try:
            df_map = pd.read_csv(csv_path, header=None, names=['id', 'name'])
            for _, row in df_map.iterrows():
                global_mapping[str(row['id']).strip()] = str(row['name']).strip()
        except:
            pass

    def map_from_csv(cls_id):
        cls_str = str(cls_id)
        name = global_mapping.get(cls_str)
        if name:
            return f"{name} ({cls_str})"
        return cls_str
    df['predicted_class'] = df['predicted_class'].apply(map_from_csv)

    # Agnostic NMS to remove overlapping duplicate detections
    def compute_iou(row1, row2):
        x_left = max(row1['x1'], row2['x1'])
        y_top = max(row1['y1'], row2['y1'])
        x_right = min(row1['x2'], row2['x2'])
        y_bottom = min(row1['y2'], row2['y2'])
        if x_right < x_left or y_bottom < y_top: return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (row1['x2'] - row1['x1']) * (row1['y2'] - row1['y1'])
        area2 = (row2['x2'] - row2['x1']) * (row2['y2'] - row2['y1'])
        return intersection / float(area1 + area2 - intersection)

    if 'class_confidence' in df.columns:
        df = df.sort_values('class_confidence', ascending=False).reset_index(drop=True)

    keep_indices = []
    for i in range(len(df)):
        keep = True
        for j in keep_indices:
            if compute_iou(df.iloc[i], df.iloc[j]) > 0.6:
                keep = False
                break
        if keep: keep_indices.append(i)
    df = df.iloc[keep_indices].reset_index(drop=True)

    # Filter out low-confidence classifications (signs, posters, partial items)
    CONF_THRESHOLD = 0.45
    if 'class_confidence' in df.columns:
        df = df[df['class_confidence'] >= CONF_THRESHOLD].reset_index(drop=True)

    # Filter out bottom-edge cut-off items (half-visible bottom shelf)
    import cv2 as _cv2
    _img = _cv2.imread(image_path)
    if _img is not None and not df.empty:
        _img_h = _img.shape[0]
        _heights = (df['y2'] - df['y1'])
        _median_h = _heights.median()
        _bottom_cutoff = (df['y2'] >= _img_h - 10) & (_heights < _median_h * 0.5)
        df = df[~_bottom_cutoff].reset_index(drop=True)

    # 3. Planogram grouping
    img_h = int(df['y2'].max()) + 50
    shelf_lines = detect_shelf_lines(df, img_h)
    df_shelved = assign_shelves(df, shelf_lines)
    
    # 4. Evaluate using Hybrid Logic (Golden Image + Heuristics)
    golden_schema_path = Path(schemas_dir) / "golden_schema.json"
    expected_schema = None
    if golden_schema_path.exists():
        try:
            expected_schema = load_schema(str(golden_schema_path))
        except Exception as e:
            print(f"Error loading golden schema: {e}")
            
    results = evaluate_hybrid_shelves(df_shelved, expected_schema)
    
    # 5. Generate planogram image with cell position tracking
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _img = cv2.imread(image_path)
    _img_h, _img_w = (_img.shape[:2]) if _img is not None else (img_h, int(df['x2'].max()) + 50)
    base_name = Path(image_path).stem
    plan_path = str(plan_folder / f"{base_name}_planogram.png")

    fig, planogram_cells = generate_planogram(
        df_shelved, _img_h, _img_w,
        image_path=image_path,
        output_path=plan_path,
        show_images=True,
        title="Planogram Analysis",
    )
    plt.close(fig)

    # Build lookup: original bbox → planogram pixel bbox
    # Use int() on both sides to avoid float64 vs int mismatch in dict key lookup
    bbox_map = {}
    for cell in planogram_cells:
        ob = cell.get('orig_bbox', {})
        pb = cell.get('planogram_bbox')
        if pb and ob:
            key = (int(ob['x1']), int(ob['y1']), int(ob['x2']), int(ob['y2']))
            bbox_map[key] = pb

    # Save original-image bbox annotations for compliance image click interaction (must be BEFORE bbox remapping)
    compliance_annotations = []
    for item in results.get('correct_items', []):
        if item.get('bbox'):
            compliance_annotations.append({'item': {k: v for k, v in item.items()}, 'type': 'correct'})
    for item in results.get('misplaced_items', []):
        if item.get('bbox'):
            compliance_annotations.append({'item': {k: v for k, v in item.items()}, 'type': 'misplaced'})
    for item in results.get('gap_detections', []):
        if item.get('bbox'):
            compliance_annotations.append({'item': {k: v for k, v in item.items()}, 'type': 'gap'})

    # Generate compliance image: draw colored boxes on original photo (must be BEFORE bbox remapping)
    comp_img = _img.copy() if _img is not None else None
    if comp_img is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 4
        text_thickness = 2

        # Draw correct items - green
        for item in results.get('correct_items', []):
            b = item.get('bbox')
            if b:
                cv2.rectangle(comp_img, (b['x1'], b['y1']), (b['x2'], b['y2']), (0, 220, 0), thickness)

        # Draw misplaced items - red
        for item in results.get('misplaced_items', []):
            b = item.get('bbox')
            if b:
                cv2.rectangle(comp_img, (b['x1'], b['y1']), (b['x2'], b['y2']), (0, 0, 220), thickness)

        # Draw gap detections - yellow with "BOS" label
        for item in results.get('gap_detections', []):
            b = item.get('bbox')
            if b:
                cv2.rectangle(comp_img, (b['x1'], b['y1']), (b['x2'], b['y2']), (0, 220, 220), thickness)
                label = "BOS"
                text_x = b['x1'] + 4
                text_y = b['y1'] + 14
                cv2.putText(comp_img, label, (text_x, text_y), font, font_scale, (0, 0, 0), text_thickness + 1, cv2.LINE_AA)
                cv2.putText(comp_img, label, (text_x, text_y), font, font_scale, (0, 220, 220), text_thickness, cv2.LINE_AA)

        comp_folder = output_folder / "compliance"
        comp_folder.mkdir(parents=True, exist_ok=True)
        comp_path = str(comp_folder / f"{base_name}_compliance.jpg")
        cv2.imwrite(comp_path, comp_img)
        results['compliance_image_url'] = f"/outputs/compliance/{base_name}_compliance.jpg"
        results['compliance_annotations'] = compliance_annotations

    # Replace bboxes in comparator results with planogram pixel positions
    for category in ('correct_items', 'misplaced_items', 'unexpected_items'):
        for item in results.get(category, []):
            b = item.get('bbox')
            if b:
                key = (int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2']))
                if key in bbox_map:
                    item['bbox'] = bbox_map[key]

    # Gap bboxes are synthetic (not real detections) and have no planogram cells,
    # so remove them to prevent false click matches on the planogram image.
    for item in results.get('gap_detections', []):
        item.pop('bbox', None)

    results['image_url'] = f"/outputs/planogram/{base_name}_planogram.png"
    results['status'] = 'success'
    results['message'] = "Analysis completed successfully."

    return results

def set_reference_image(image_path: str, schemas_dir: str, output_folder: Path) -> dict:
    """Processes an image and saves the exact detected layout as the Golden Image."""
    initialize_models()
    
    cls_folder = output_folder / "classification"
    det_folder = output_folder / "detection"
    plan_folder = output_folder / "planogram"
    for folder in [cls_folder, det_folder, plan_folder]:
        folder.mkdir(parents=True, exist_ok=True)
        
    df, timing = process_image(
        image_path,
        _detection_model,
        _classification_model,
        _ref_embeddings,
        _ref_class_names,
        cls_folder,
        det_folder,
        plan_folder
    )
    
    if df.empty:
        return {"status": "error", "message": "No products detected on reference image."}
        
    csv_path = ROOT_DIR / "datasets" / "migros_dataset_v6" / "Annotations" / "SDP_Product&ID_Dataset_fix.csv"
    global_mapping = {}
    if csv_path.exists():
        try:
            df_map = pd.read_csv(csv_path, header=None, names=['id', 'name'])
            for _, row in df_map.iterrows():
                global_mapping[str(row['id']).strip()] = str(row['name']).strip()
        except:
            pass

    def map_from_csv(cls_id):
        cls_str = str(cls_id)
        name = global_mapping.get(cls_str)
        if name:
            return f"{name} ({cls_str})"
        return cls_str
    df['predicted_class'] = df['predicted_class'].apply(map_from_csv)

    # Agnostic NMS to remove overlapping duplicate detections
    def compute_iou(row1, row2):
        x_left = max(row1['x1'], row2['x1'])
        y_top = max(row1['y1'], row2['y1'])
        x_right = min(row1['x2'], row2['x2'])
        y_bottom = min(row1['y2'], row2['y2'])
        if x_right < x_left or y_bottom < y_top: return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (row1['x2'] - row1['x1']) * (row1['y2'] - row1['y1'])
        area2 = (row2['x2'] - row2['x1']) * (row2['y2'] - row2['y1'])
        return intersection / float(area1 + area2 - intersection)

    if 'class_confidence' in df.columns:
        df = df.sort_values('class_confidence', ascending=False).reset_index(drop=True)
    keep_indices = []
    for i in range(len(df)):
        keep = True
        for j in keep_indices:
            if compute_iou(df.iloc[i], df.iloc[j]) > 0.6:
                keep = False
                break
        if keep: keep_indices.append(i)
    df = df.iloc[keep_indices].reset_index(drop=True)

    # Filter out low-confidence classifications
    CONF_THRESHOLD = 0.45
    if 'class_confidence' in df.columns:
        df = df[df['class_confidence'] >= CONF_THRESHOLD].reset_index(drop=True)

    img_h = int(df['y2'].max()) + 50
    shelf_lines = detect_shelf_lines(df, img_h)
    df_shelved = assign_shelves(df, shelf_lines)
    
    schema = generate_schema_from_df(df_shelved)
    
    import json
    schemas_path = Path(schemas_dir)
    schemas_path.mkdir(parents=True, exist_ok=True)
    out_file = schemas_path / "golden_schema.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    # Save a copy of the reference image for UI display
    import cv2 as _cv2_ref
    _ref_dir = Path(output_folder) / "reference"
    _ref_dir.mkdir(parents=True, exist_ok=True)
    _ref_img_data = _cv2_ref.imread(image_path)
    if _ref_img_data is not None:
        _cv2_ref.imwrite(str(_ref_dir / "reference_image.jpg"), _ref_img_data)

    return {"status": "success", "message": "Reference saved successfully."}

