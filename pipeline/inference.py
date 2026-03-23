import sys
from pathlib import Path
import pandas as pd

# Add root project dir to path to import demo and planogram modules
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from demo import load_classification_model, run_detection, process_image, YOLO, CLASSIFICATION_CHECKPOINT, REFERENCE_DB_PATH, DETECTION_WEIGHTS
from planogram import detect_shelf_lines, assign_shelves
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
        
    csv_path = Path("C:/Users/Casper/Desktop/grocery_project/datasets/migros_dataset_v6/Annotations/SDP_Product&ID_Dataset_fix.csv")
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
        name = global_mapping.get(cls_str, "Ürün")
        return f"{name} ({cls_str})"
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
    
    # 5. Visual Compliance Overlay
    import cv2
    img = cv2.imread(image_path)
    if img is not None:
        for item in results.get("correct_items", []):
            b = item.get("bbox")
            if b: cv2.rectangle(img, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 255, 0), 4)

        for item in results.get("misplaced_items", []):
            b = item.get("bbox")
            if b: cv2.rectangle(img, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 0, 255), 4)

        for item in results.get("unexpected_items", []):
            pass

        # Draw yellow rectangles for physical gaps (empty spaces)
        for item in results.get("gap_detections", []):
            b = item.get("bbox")
            if b:
                cv2.rectangle(img, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 255, 255), 4)
                cv2.putText(img, "BOS", (b["x1"]+5, b["y1"]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        base_name = Path(image_path).stem
        out_path = plan_folder / f"{base_name}_compliance.jpg"
        cv2.imwrite(str(out_path), img)
        results['image_url'] = f"/outputs/planogram/{base_name}_compliance.jpg"
    else:
        results['image_url'] = f"/outputs/planogram/{Path(image_path).stem}_planogram.png"
    
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
        
    csv_path = Path("C:/Users/Casper/Desktop/grocery_project/datasets/migros_dataset_v6/Annotations/SDP_Product&ID_Dataset_fix.csv")
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
        name = global_mapping.get(cls_str, "Ürün")
        return f"{name} ({cls_str})"
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
        
    return {"status": "success", "message": "Reference saved successfully."}

