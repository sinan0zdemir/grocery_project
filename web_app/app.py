# uv run uvicorn web_app.app:app --reload

import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uuid

# Define base directory
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Grocery Planogram Analyzer")

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Define upload folder
UPLOAD_DIR = BASE_DIR.parent / "demo_output" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
        
    import sys
    sys.path.insert(0, str(BASE_DIR.parent))
    from pipeline.inference import run_analysis
    
    # Define paths
    output_folder = BASE_DIR.parent / "demo_output"
    schemas_dir = BASE_DIR.parent / "planogram" / "schemas"
    
    # Run the ML pipeline
    try:
        results = run_analysis(str(file_path), str(schemas_dir), output_folder)
        # Update image URL to point to a mounted static folder or endpoint
        # The frontend expects 'image_url' to be the visual planogram
        base_name = file_path.stem
        # Ensure we have a route to serve these files or copy them to static
        # Assuming we serve output via a new mount
        results['image_url'] = f"/outputs/planogram/{base_name}_planogram.png"
    except Exception as e:
        results = {"status": "error", "message": f"Pipeline failed: {str(e)}"}
        
    return JSONResponse(content=results)

# Mount outputs so the frontend can display the processed images
app.mount("/outputs", StaticFiles(directory=str(BASE_DIR.parent / "demo_output")), name="outputs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
