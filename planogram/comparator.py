import pandas as pd
from typing import Dict, List
import json

def load_schema(schema_path: str) -> dict:
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_shelves(detected_df: pd.DataFrame, expected_schema: dict) -> dict:
    """
    Compares the shelf layout detected by `planogram.py` against the expected JSON schema.
    
    Args:
        detected_df: DataFrame output of `assign_shelves` in `planogram.py`. 
                     Should have ['shelf', 'predicted_class', 'x1', ...] columns.
                     Assumes products have been sorted horizontally.
        expected_schema: Dictionary containing the desired layout, e.g. 
                         {'rows': [['item1', 'item2'], ['item3']]}
                         
    Returns:
        dict: Summary of missing, misplaced, and correctly placed items.
    """
    
    # Extract expected rows (0 is top shelf, 1 is next shelf down, etc)
    expected_rows = expected_schema.get('rows', [])
    num_expected_shelves = len(expected_rows)
    
    detected_df = detected_df.copy()
    
    # Get detected shelves
    # Note: shelf IDs in the dataframe might be [0, 1, 2, ...].
    shelf_ids = sorted(detected_df['shelf'].unique())
    num_detected_shelves = len(shelf_ids)
    
    results = {
        "compliance_score": 100.0,
        "total_items": len(detected_df),
        "correct_items": [],
        "missing_items": [],
        "misplaced_items": [],
        "unexpected_items": [],
        "category_score": 0
    }
    
    correct_count = 0
    total_expected = sum(len(row) for row in expected_rows)
    
    # For each shelf, compare using count-based matching (not brittle 1:1 position alignment).
    # This is far more robust: it counts how many of each product type are expected vs detected.
    for shelf_idx in range(max(num_expected_shelves, num_detected_shelves)):
        expected_items = expected_rows[shelf_idx] if shelf_idx < num_expected_shelves else []
        
        # Get detected items for this shelf, sorted by x1
        if shelf_idx < num_detected_shelves:
            actual_shelf_id = shelf_ids[shelf_idx]
            detected_items_df = detected_df[detected_df['shelf'] == actual_shelf_id].sort_values('x1')
            
            detected_items = []
            detected_rows_list = []
            
            for _, row in detected_items_df.iterrows():
                detected_items.append(row['predicted_class'])
                detected_rows_list.append(row)
        else:
            detected_items = []
            detected_rows_list = []
        
        def get_base_name(name):
            if not name: return None
            return name.rsplit('(', 1)[0].strip()
            
        # --- Count-based matching per shelf ---
        # Build expected counts (by base name)
        from collections import Counter
        expected_counts = Counter(get_base_name(e) for e in expected_items)
        detected_base_list = [get_base_name(d) for d in detected_items]
        detected_counts = Counter(detected_base_list)
        
        # 1) Mark matched (correct) items
        matched_indices = set()
        for base_name, exp_count in expected_counts.items():
            det_count = detected_counts.get(base_name, 0)
            matched = min(exp_count, det_count)
            correct_count += matched
            
            # Find the first N detected items with this base name and mark them correct
            found = 0
            for idx, det_base in enumerate(detected_base_list):
                if found >= matched:
                    break
                if det_base == base_name and idx not in matched_indices:
                    matched_indices.add(idx)
                    det_row = detected_rows_list[idx]
                    bbox = {"x1": int(det_row['x1']), "y1": int(det_row['y1']),
                            "x2": int(det_row['x2']), "y2": int(det_row['y2'])}
                    results["correct_items"].append({
                        "label": detected_items[idx],
                        "shelf": shelf_idx + 1,
                        "position": idx + 1,
                        "bbox": bbox
                    })
                    found += 1
            
            # 2) If expected > detected for this product => missing
            if exp_count > det_count:
                for _ in range(exp_count - det_count):
                    # Find original expected name for this base
                    orig_name = next((e for e in expected_items if get_base_name(e) == base_name), base_name)
                    results["missing_items"].append({
                        "label": orig_name,
                        "expected_shelf": shelf_idx + 1,
                        "expected_position": None
                    })
        
        # 3) Unmatched detected items => categorize properly
        # Build set of base names expected on THIS shelf specifically
        this_shelf_bases = set(get_base_name(e) for e in expected_items)
        # Build set of base names expected on OTHER shelves
        other_shelf_bases = set()
        for other_idx, other_row in enumerate(expected_rows):
            if other_idx != shelf_idx:
                other_shelf_bases.update(get_base_name(e) for e in other_row)
        
        for idx, det_item in enumerate(detected_items):
            if idx not in matched_indices:
                det_row = detected_rows_list[idx]
                bbox = {"x1": int(det_row['x1']), "y1": int(det_row['y1']),
                        "x2": int(det_row['x2']), "y2": int(det_row['y2'])}
                det_base = detected_base_list[idx]
                
                if det_base in this_shelf_bases:
                    # Extra stock of a product that BELONGS on this shelf → correct (surplus)
                    correct_count += 1
                    results["correct_items"].append({
                        "label": det_item,
                        "shelf": shelf_idx + 1,
                        "position": idx + 1,
                        "bbox": bbox
                    })
                elif det_base in other_shelf_bases:
                    # Product exists in schema but on a DIFFERENT shelf → misplaced
                    results["misplaced_items"].append({
                        "expected_label": "(Farklı rafta olmalı)",
                        "detected_label": det_item,
                        "expected_shelf": None,
                        "detected_shelf": shelf_idx + 1,
                        "position": idx + 1,
                        "bbox": bbox
                    })
                else:
                    # Product not in schema at all → unexpected
                    results["unexpected_items"].append({
                        "label": det_item,
                        "detected_shelf": shelf_idx + 1,
                        "detected_position": idx + 1,
                        "bbox": bbox
                    })
                
    # --- Physical Gap Detection ---
    # Scan each detected shelf for large spaces between products
    gap_items = []
    for shelf_idx in range(num_detected_shelves):
        actual_shelf_id = shelf_ids[shelf_idx]
        shelf_df = detected_df[detected_df['shelf'] == actual_shelf_id].sort_values('x1')
        
        if len(shelf_df) < 2:
            continue
            
        widths = (shelf_df['x2'] - shelf_df['x1']).values
        median_w = max(float(pd.Series(widths).median()), 20.0)
        
        prev_x2 = None
        for _, row in shelf_df.iterrows():
            if prev_x2 is not None:
                gap = row['x1'] - prev_x2
                if gap > 1.8 * median_w:
                    num_missing = max(1, int(round(gap / median_w)) - 1)
                    for g in range(num_missing):
                        gap_x = int(prev_x2 + (g + 1) * median_w)
                        gap_items.append({
                            "label": f"Boş Alan (Raf {shelf_idx + 1})",
                            "expected_shelf": shelf_idx + 1,
                            "expected_position": None,
                            "bbox": {"x1": gap_x, "y1": int(row['y1']), 
                                    "x2": int(gap_x + median_w), "y2": int(row['y2'])}
                        })
            prev_x2 = row['x2']
    
    results["gap_detections"] = gap_items
                
    # Calculate simple score
    if total_expected > 0:
        results["compliance_score"] = round((correct_count / total_expected) * 100, 2)
        
    # Calculate category score for Auto-Selector (Schema Selection)
    def extract_brand(name):
        if not name: return ""
        return name.rsplit('(', 1)[0].split(' ')[0]
        
    expected_brands = set(extract_brand(item) for row in expected_rows for item in row if item)
    detected_items_all = detected_df['predicted_class'].tolist()
    
    cat_score = 0
    for item in detected_items_all:
        if extract_brand(item) in expected_brands:
            cat_score += 1
    results["category_score"] = cat_score
        
    return results

# Self-test block
if __name__ == "__main__":
    schema = {
        "rows": [
            ["Coke", "Coke", "Fanta"],
            ["Sprite", "Water"]
        ]
    }
    
    # Mock dataframe
    data = [
        {'shelf': 0, 'predicted_class': 'Coke', 'x1': 10},
        {'shelf': 0, 'predicted_class': 'Fanta', 'x1': 50},  # Coke is missing here -> Mismatch
        {'shelf': 0, 'predicted_class': 'Sprite', 'x1': 90}, # Misplaced
        {'shelf': 1, 'predicted_class': 'Sprite', 'x1': 10},
        {'shelf': 1, 'predicted_class': 'Water', 'x1': 50}
    ]
    df = pd.DataFrame(data)
    
    res = compare_shelves(df, schema)
    print("Test Results:", res)
