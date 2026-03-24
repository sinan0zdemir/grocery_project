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
        # Build dict of base names expected on OTHER shelves mapping to shelf numbers
        other_shelf_dict = {}
        for other_idx, other_row in enumerate(expected_rows):
            if other_idx != shelf_idx:
                for e in other_row:
                    bn = get_base_name(e)
                    if bn not in other_shelf_dict:
                        other_shelf_dict[bn] = []
                    other_shelf_dict[bn].append(other_idx + 1)
        
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
                elif det_base in other_shelf_dict:
                    # Product exists in schema but on a DIFFERENT shelf → misplaced
                    exp_shelves = sorted(list(set(other_shelf_dict[det_base])))
                    shelf_str = " & ".join(map(str, exp_shelves))
                    results["misplaced_items"].append({
                        "expected_label": f"Belongs on Shelf {shelf_str}",
                        "detail_msg": f"Belongs on Shelf {shelf_str} compared to Golden Image",
                        "detected_label": det_item,
                        "expected_shelf": exp_shelves[0],
                        "detected_shelf": shelf_idx + 1,
                        "position": idx + 1,
                        "bbox": bbox
                    })
                else:
                    # Product not in schema at all → unexpected
                    results["unexpected_items"].append({
                        "label": det_item,
                        "detail_msg": "Unexpected item compared to Golden Image",
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

def evaluate_shelves_heuristic(detected_df: pd.DataFrame) -> dict:
    """
    Evaluates shelves based on heuristic rules (no planogram schema).
    - Missing (Yellow): Physical gaps between products.
    - Misplaced (Red): An anomaly, a product stuck in the middle of a different product's block.
    - Correct (Green): All other items in clusters.
    """
    detected_df = detected_df.copy()
    shelf_ids = sorted(detected_df['shelf'].unique())
    num_detected_shelves = len(shelf_ids)
    
    results = {
        "compliance_score": 100.0, # Visual display only, maybe replace later
        "total_items": len(detected_df),
        "correct_items": [],
        "misplaced_items": [],
        "gap_detections": [],
        "unexpected_items": [], # Keep empty for UI compatibility
        "missing_items": [] # Keep empty for UI compatibility
    }
    
    def get_base_name(name):
        return name.rsplit('(', 1)[0].strip() if name else ""

    # Physical Gap Detection
    gap_items = []
    
    for shelf_idx in range(num_detected_shelves):
        actual_shelf_id = shelf_ids[shelf_idx]
        shelf_df = detected_df[detected_df['shelf'] == actual_shelf_id].sort_values('x1')
        
        if len(shelf_df) < 2:
            continue
            
        widths = (shelf_df['x2'] - shelf_df['x1']).values
        median_w = max(float(pd.Series(widths).median()), 20.0)
        
        # Check gaps between products
        prev_x2 = None
        for _, row in shelf_df.iterrows():
            if prev_x2 is not None:
                gap = row['x1'] - prev_x2
                if gap > 0.6 * median_w:
                    num_missing = max(1, int(round(gap / median_w)))
                    for g in range(num_missing):
                        gap_w = min(median_w, gap / num_missing)
                        gap_x = int(prev_x2 + g * gap_w)
                        gap_items.append({
                            "label": f"Empty Space",
                            "expected_shelf": shelf_idx + 1,
                            "bbox": {"x1": gap_x, "y1": int(row['y1']), 
                                    "x2": int(gap_x + gap_w), "y2": int(row['y2'])}
                        })
            if prev_x2 is None:
                prev_x2 = row['x2']
            else:
                prev_x2 = max(prev_x2, row['x2'])
            
    results["gap_detections"] = gap_items
    
    # Misplaced and Correct Logic (Clustering & Anomaly)
    for shelf_idx in range(num_detected_shelves):
        actual_shelf_id = shelf_ids[shelf_idx]
        shelf_df = detected_df[detected_df['shelf'] == actual_shelf_id].sort_values('x1')
        
        items = []
        for idx, row in shelf_df.iterrows():
            items.append({
                "label": row['predicted_class'],
                "base_name": get_base_name(row['predicted_class']),
                "row": row,
                "bbox": {"x1": int(row['x1']), "y1": int(row['y1']), 
                         "x2": int(row['x2']), "y2": int(row['y2'])}
            })
            
        n = len(items)
        if n == 0:
            continue
            
        for i in range(n):
            current = items[i]
            is_misplaced = False
            target_brand = get_base_name(shelf_df.iloc[i]['predicted_class'])
            left_brand = get_base_name(shelf_df.iloc[i-1]['predicted_class']) if i > 0 else None
            right_brand = get_base_name(shelf_df.iloc[i+1]['predicted_class']) if i < n-1 else None
            
            has_identical_neighbor = (left_brand == target_brand) or (right_brand == target_brand)
            
            if not has_identical_neighbor:
                # Isolated item. Let's see if it's flanked on both sides by foreign brands
                if left_brand is not None and right_brand is not None:
                    is_misplaced = True
                    detail_msg = f"1 {target_brand} item isolated between {left_brand} and {right_brand}"
                else:
                    # It's at the edge (or alone on the shelf). Let's check the majority brand of the shelf.
                    majority_brand = shelf_df['predicted_class'].apply(get_base_name).mode()[0]
                    if target_brand != majority_brand:
                        is_misplaced = True
                        if left_brand is None and right_brand is None:
                            detail_msg = f"Foreign object ({target_brand}). Shelf majority is {majority_brand}"
                        else:
                            neighbor = left_brand if left_brand else right_brand
                            detail_msg = f"Foreign object at edge next to {neighbor}. Shelf majority is {majority_brand}"
            
            if is_misplaced:
                results["misplaced_items"].append({
                    "detected_label": current['label'],
                    "expected_label": majority_brand if 'majority_brand' in locals() else left_brand,
                    "expected_shelf": shelf_idx + 1,
                    "detected_shelf": shelf_idx + 1,
                    "bbox": current["bbox"],
                    "detail_msg": detail_msg
                })
            else:
                results["correct_items"].append({
                    "label": current['label'],
                    "shelf": shelf_idx + 1,
                    "position": i + 1,
                    "bbox": current["bbox"]
                })
                
    return results

def generate_schema_from_df(detected_df: pd.DataFrame) -> dict:
    """Creates a JSON schema representation of the currently detected shelves."""
    detected_df = detected_df.copy()
    shelf_ids = sorted(detected_df['shelf'].unique())
    
    rows = []
    for shelf_idx in range(len(shelf_ids)):
        actual_shelf_id = shelf_ids[shelf_idx]
        shelf_df = detected_df[detected_df['shelf'] == actual_shelf_id].sort_values('x1')
        rows.append(shelf_df['predicted_class'].tolist())
        
    return {"rows": rows}

def evaluate_hybrid_shelves(detected_df: pd.DataFrame, expected_schema: dict = None) -> dict:
    """
    Hybrid evaluation:
    If expected_schema is None -> Use purely heuristic logic.
    If expected_schema is provided -> Golden Image is the absolute truth for Correct/Missing/Misplaced.
                                      Heuristic logic is strictly DISABLED, except for finding physical gaps.
    """
    heuristic_res = evaluate_shelves_heuristic(detected_df)
    
    if not expected_schema:
        return heuristic_res
        
    schema_res = compare_shelves(detected_df, expected_schema)
    
    # 1. Provide physical empty-space gap detections from the visual heuristics
    schema_res["gap_detections"] = heuristic_res.get("gap_detections", [])
    
    return schema_res

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
