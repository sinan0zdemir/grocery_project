import sys
from pathlib import Path
import pandas as pd

# Add root project dir to path to import demo and planogram modules
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from demo import load_classification_model, run_detection, process_image, YOLO, CLASSIFICATION_CHECKPOINT, REFERENCE_DB_PATH, DETECTION_WEIGHTS
from planogram import detect_shelf_lines, assign_shelves
from comparator import compare_shelves, load_schema
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
        
    # 3. Planogram grouping
    img_h = 2000 # dummy default, normally process_image has access to it but returns only df. We can estimate.
    # We will estimate img_h from df y2 max
    img_h = int(df['y2'].max()) + 50
    shelf_lines = detect_shelf_lines(df, img_h)
    df_shelved = assign_shelves(df, shelf_lines)
    
    # 4. JSON Compare against all schemas to find the best match
    best_results = None
    best_score = -1
    
    for schema in schemas:
        results = compare_shelves(df_shelved, schema)
        if results.get("category_score", 0) > best_score:
            best_score = results.get("category_score", 0)
            best_results = results
            
    results = best_results if best_results is not None else compare_shelves(df_shelved, schemas[0])
    
    # 5. Visual Compliance Overlay
    import cv2
    img = cv2.imread(image_path)
    if img is not None:
        for item in results.get("correct_items", []):
            b = item.get("bbox")
            if b: cv2.rectangle(img, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 255, 0), 4)

        for item in results.get("misplaced_items", []):
            b = item.get("bbox")
            if b: cv2.rectangle(img, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 165, 255), 4)

        for item in results.get("unexpected_items", []):
            b = item.get("bbox")
            if b: cv2.rectangle(img, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 0, 255), 4)

        # Draw purple rectangles for physical gaps (empty spaces)
        for item in results.get("gap_detections", []):
            b = item.get("bbox")
            if b:
                cv2.rectangle(img, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (255, 0, 255), 4)
                cv2.putText(img, "BOS", (b["x1"]+5, b["y1"]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
        base_name = Path(image_path).stem
        out_path = plan_folder / f"{base_name}_compliance.jpg"
        cv2.imwrite(str(out_path), img)
        results['image_url'] = f"/outputs/planogram/{base_name}_compliance.jpg"
    else:
        results['image_url'] = f"/outputs/planogram/{Path(image_path).stem}_planogram.png"
    
    results['status'] = 'success'
    results['message'] = "Analysis completed successfully."
    
    return results
