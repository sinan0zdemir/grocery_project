# uv run uvicorn web_app.app:app --reload

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uuid

# Define base directory
BASE_DIR = Path(__file__).resolve().parent

# Ensure project root is on sys.path so pipeline imports work
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

from pipeline.inference import initialize_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load YOLO and ArcFace models into memory at startup."""
    print("🚀 Pre-loading ML models into memory...")
    initialize_models()
    print("✅ Models loaded and ready.")
    yield
    # Cleanup (nothing needed – models live until process exits)


app = FastAPI(title="Grocery Planogram Analyzer", lifespan=lifespan)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Create all output directories at startup
OUTPUT_DIR = BASE_DIR.parent / "demo_output"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
for _d in [UPLOAD_DIR, OUTPUT_DIR / "classification", OUTPUT_DIR / "detection", OUTPUT_DIR / "planogram", OUTPUT_DIR / "compliance"]:
    _d.mkdir(parents=True, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main UI dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """API endpoint to receive an image and return planogram compliance."""
    # 1. Save uploaded image
    file_ext = Path(file.filename).suffix
    unique_id = str(uuid.uuid4())[:8]
    file_name = f"upload_{unique_id}{file_ext}"
    file_path = UPLOAD_DIR / file_name
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    from pipeline.inference import run_analysis
    
    # Define paths
    output_folder = BASE_DIR.parent / "demo_output"
    schemas_dir = BASE_DIR.parent / "planogram" / "schemas"
    
    # Run the ML pipeline
    try:
        results = run_analysis(str(file_path), str(schemas_dir), output_folder)
        # image_url is set by run_analysis to the planogram image,
        # with bbox coordinates mapped to planogram pixel positions.
    except Exception as e:
        results = {"status": "error", "message": f"Pipeline failed: {str(e)}"}
        
    return JSONResponse(content=results)

@app.post("/api/set_reference")
async def set_reference(file: UploadFile = File(...), name: str = Form(None)):
    """Save the uploaded image as a new reference shelf and generate its reference schema."""
    file_ext = Path(file.filename).suffix
    ref_id = str(uuid.uuid4())[:8]
    file_name = f"ref_{ref_id}{file_ext}"
    file_path = UPLOAD_DIR / file_name
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    from pipeline.inference import set_reference_image
    
    # Define paths
    output_folder = BASE_DIR.parent / "demo_output"
    schemas_dir = BASE_DIR.parent / "planogram" / "schemas"
    
    try:
        results = set_reference_image(str(file_path), str(schemas_dir), output_folder)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": f"Failed to set reference: {str(e)}"})
    
    if results.get("status") != "success":
        return JSONResponse(content=results)
    
    # Copy the uploaded image to the references folder for display
    ref_image_name = f"ref_{ref_id}{file_ext}"
    ref_image_path = REFERENCES_DIR / ref_image_name
    shutil.copy2(str(file_path), str(ref_image_path))
    
    # Copy the generated reference_schema to a per-reference schema file
    ref_source_path = Path(schemas_dir) / "reference_schema.json"
    ref_schema_name = f"schema_{ref_id}.json"
    ref_schema_path = REFERENCES_DIR / ref_schema_name
    if ref_source_path.exists():
        shutil.copy2(str(ref_source_path), str(ref_schema_path))
    
    # Auto-generate name if not provided
    if not name:
        name = f"Shelf Reference ({datetime.now().strftime('%d/%m/%Y %H:%M')})"
    
    # Deactivate all existing references, activate this new one
    refs = load_references_meta()
    for r in refs:
        r["active"] = False
    
    new_ref = {
        "id": ref_id,
        "name": name,
        "image_url": f"/outputs/references/{ref_image_name}",
        "schema_path": ref_schema_name,
        "created_at": datetime.now().isoformat(),
        "active": True
    }
    refs.append(new_ref)
    save_references_meta(refs)
    
    return JSONResponse(content={
        "status": "success",
        "message": "Reference saved successfully.",
        "reference": new_ref
    })

@app.get("/api/list_references")
async def list_references():
    """Return all saved references."""
    refs = load_references_meta()
    return JSONResponse(content={"references": refs})

@app.post("/api/activate_reference")
async def activate_reference(ref_id: str = Form(...)):
    """Set a specific saved reference as the active reference schema."""
    refs = load_references_meta()
    target = None
    for r in refs:
        if r["id"] == ref_id:
            r["active"] = True
            target = r
        else:
            r["active"] = False
    
    if target is None:
        return JSONResponse(content={"status": "error", "message": "Reference not found."})
    
    # Copy target schema to reference_schema.json
    schemas_dir = BASE_DIR.parent / "planogram" / "schemas"
    schemas_dir.mkdir(parents=True, exist_ok=True)
    src_schema = REFERENCES_DIR / target["schema_path"]
    dst_schema = schemas_dir / "reference_schema.json"
    
    if src_schema.exists():
        shutil.copy2(str(src_schema), str(dst_schema))
    else:
        return JSONResponse(content={"status": "error", "message": "Schema file missing for this reference."})
    
    save_references_meta(refs)
    return JSONResponse(content={"status": "success", "message": f"Reference '{target['name']}' activated.", "reference": target})

@app.post("/api/clear_reference")
async def clear_reference(ref_id: str = Form(None)):
    """Delete a specific reference by ID, or clear the active one if no ID is given."""
    refs = load_references_meta()
    schemas_dir = BASE_DIR.parent / "planogram" / "schemas"
    golden = schemas_dir / "golden_schema.json"
    if golden.exists():
        golden.unlink()
    ref_img = BASE_DIR.parent / "demo_output" / "reference" / "reference_image.jpg"
    if ref_img.exists():
        ref_img.unlink()
    return JSONResponse(content={"status": "success", "message": "Reference cleared."})

@app.get("/api/check_reference")
async def check_reference():
    """Check if a reference schema currently exists and return info about the active reference."""
    schemas_dir = BASE_DIR.parent / "planogram" / "schemas"
    golden = schemas_dir / "golden_schema.json"
    ref_image_url = None
    ref_img = BASE_DIR.parent / "demo_output" / "reference" / "reference_image.jpg"
    if ref_img.exists():
        ref_image_url = "/outputs/reference/reference_image.jpg"
    return JSONResponse(content={"has_reference": golden.exists(), "ref_image_url": ref_image_url})

# Mount outputs so the frontend can display the processed images
app.mount("/outputs", StaticFiles(directory=str(BASE_DIR.parent / "demo_output")), name="outputs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
