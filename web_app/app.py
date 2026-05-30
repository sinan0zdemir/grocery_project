# uv run uvicorn web_app.app:app --reload

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware


class NoCacheHTMLMiddleware(BaseHTTPMiddleware):
    """Force HTML responses to revalidate so users always see the latest layout."""
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        ctype = response.headers.get("content-type", "")
        if "text/html" in ctype:
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        return response
import uuid

# Define base directory
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Grocery Planogram Analyzer")
app.add_middleware(NoCacheHTMLMiddleware)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Create all output directories at startup
OUTPUT_DIR = BASE_DIR.parent / "demo_output"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
REFERENCES_DIR = OUTPUT_DIR / "references"
SCHEMAS_DIR = BASE_DIR.parent / "planogram" / "schemas"
REFERENCES_INDEX = REFERENCES_DIR / "references.json"
GOLDEN_SCHEMA = SCHEMAS_DIR / "golden_schema.json"
ACTIVE_REF_IMG = OUTPUT_DIR / "reference" / "reference_image.jpg"

for _d in [UPLOAD_DIR, OUTPUT_DIR / "classification", OUTPUT_DIR / "detection",
           OUTPUT_DIR / "planogram", OUTPUT_DIR / "compliance",
           REFERENCES_DIR, OUTPUT_DIR / "reference", SCHEMAS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ---------- References Helpers ----------
def _load_refs():
    if not REFERENCES_INDEX.exists():
        return []
    try:
        with open(REFERENCES_INDEX, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_refs(refs):
    with open(REFERENCES_INDEX, "w", encoding="utf-8") as f:
        json.dump(refs, f, ensure_ascii=False, indent=2)


def _active_ref():
    for r in _load_refs():
        if r.get("active"):
            return r
    return None


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main UI dashboard."""
    css_path = BASE_DIR / "static" / "css" / "style.css"
    js_path = BASE_DIR / "static" / "js" / "main.js"
    asset_version = str(int(max(
        css_path.stat().st_mtime if css_path.exists() else 0,
        js_path.stat().st_mtime if js_path.exists() else 0,
    )))
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"asset_version": asset_version},
    )


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """API endpoint to receive an image and return planogram compliance."""
    file_ext = Path(file.filename).suffix
    unique_id = str(uuid.uuid4())[:8]
    file_name = f"upload_{unique_id}{file_ext}"
    file_path = UPLOAD_DIR / file_name

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    import sys
    sys.path.insert(0, str(BASE_DIR.parent))
    from pipeline.inference import run_analysis

    output_folder = BASE_DIR.parent / "demo_output"
    schemas_dir = BASE_DIR.parent / "planogram" / "schemas"

    try:
        results = run_analysis(str(file_path), str(schemas_dir), output_folder)
    except Exception as e:
        results = {"status": "error", "message": f"Pipeline failed: {str(e)}"}

    return JSONResponse(content=results)


# ---------- New References CRUD ----------
@app.get("/api/references")
async def list_references():
    """List all stored references."""
    refs = _load_refs()
    refs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return JSONResponse(content={"references": refs})


@app.post("/api/references")
async def create_reference(file: UploadFile = File(...), name: str = None):
    """Create a new reference: process the image, save the image + schema + metadata."""
    file_ext = Path(file.filename).suffix or ".jpg"
    ref_id = str(uuid.uuid4())[:8]

    ref_image_name = f"ref_{ref_id}{file_ext}"
    ref_image_path = REFERENCES_DIR / ref_image_name
    with open(ref_image_path, "wb") as buffer:
        buffer.write(await file.read())

    import sys
    if str(BASE_DIR.parent) not in sys.path:
        sys.path.insert(0, str(BASE_DIR.parent))
    from pipeline.inference import build_schema_for_image

    schema_filename = f"schema_{ref_id}.json"
    schema_full_path = REFERENCES_DIR / schema_filename

    try:
        result = build_schema_for_image(
            str(ref_image_path),
            str(schema_full_path),
            BASE_DIR.parent / "demo_output",
        )
        if result.get("status") != "success":
            ref_image_path.unlink(missing_ok=True)
            return JSONResponse(
                content={"status": "error", "message": result.get("message", "Failed to build schema.")},
                status_code=400,
            )
    except Exception as e:
        ref_image_path.unlink(missing_ok=True)
        return JSONResponse(
            content={"status": "error", "message": f"Failed to build schema: {str(e)}"},
            status_code=500,
        )

    now = datetime.now()
    display_name = name or f"Reference ({now.strftime('%d/%m/%Y %H:%M')})"

    refs = _load_refs()
    new_ref = {
        "id": ref_id,
        "name": display_name,
        "image_url": f"/outputs/references/{ref_image_name}",
        "schema_path": schema_filename,
        "created_at": now.isoformat(),
        "active": False,
    }
    refs.append(new_ref)
    _save_refs(refs)

    return JSONResponse(content={"status": "success", "reference": new_ref})


@app.post("/api/references/{ref_id}/activate")
async def activate_reference(ref_id: str):
    """Activate a reference: copy its schema to golden_schema.json and image to active slot."""
    refs = _load_refs()
    target = next((r for r in refs if r.get("id") == ref_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Reference not found.")

    src_schema = REFERENCES_DIR / target["schema_path"]
    if not src_schema.exists():
        raise HTTPException(status_code=400, detail="Schema file missing for this reference.")

    SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_schema, GOLDEN_SCHEMA)

    # Copy image into active reference slot for preview
    ref_image_relpath = target.get("image_url", "").replace("/outputs/", "")
    src_image = OUTPUT_DIR / ref_image_relpath
    if src_image.exists():
        ACTIVE_REF_IMG.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_image, ACTIVE_REF_IMG)

    for r in refs:
        r["active"] = (r.get("id") == ref_id)
    _save_refs(refs)

    return JSONResponse(content={"status": "success", "reference": target})


@app.post("/api/references/deactivate")
async def deactivate_all_references():
    """Deactivate all references — falls back to heuristic-only detection."""
    refs = _load_refs()
    for r in refs:
        r["active"] = False
    _save_refs(refs)
    if GOLDEN_SCHEMA.exists():
        GOLDEN_SCHEMA.unlink()
    if ACTIVE_REF_IMG.exists():
        ACTIVE_REF_IMG.unlink()
    return JSONResponse(content={"status": "success"})


@app.delete("/api/references/{ref_id}")
async def delete_reference(ref_id: str):
    """Delete a reference (image + schema + index entry). If active, also clears golden schema."""
    refs = _load_refs()
    target = next((r for r in refs if r.get("id") == ref_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Reference not found.")

    schema_file = REFERENCES_DIR / target["schema_path"]
    schema_file.unlink(missing_ok=True)

    image_relpath = target.get("image_url", "").replace("/outputs/", "")
    image_file = OUTPUT_DIR / image_relpath
    image_file.unlink(missing_ok=True)

    was_active = target.get("active", False)
    refs = [r for r in refs if r.get("id") != ref_id]
    _save_refs(refs)

    if was_active:
        if GOLDEN_SCHEMA.exists():
            GOLDEN_SCHEMA.unlink()
        if ACTIVE_REF_IMG.exists():
            ACTIVE_REF_IMG.unlink()

    return JSONResponse(content={"status": "success"})


@app.patch("/api/references/{ref_id}")
async def rename_reference(ref_id: str, payload: dict):
    """Rename a reference."""
    refs = _load_refs()
    target = next((r for r in refs if r.get("id") == ref_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Reference not found.")
    new_name = (payload.get("name") or "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")
    target["name"] = new_name
    _save_refs(refs)
    return JSONResponse(content={"status": "success", "reference": target})


@app.get("/api/check_reference")
async def check_reference():
    """Check if a golden schema currently exists (legacy endpoint kept for dashboard)."""
    active = _active_ref()
    ref_image_url = None
    if ACTIVE_REF_IMG.exists():
        ref_image_url = "/outputs/reference/reference_image.jpg"
    elif active:
        ref_image_url = active.get("image_url")
    return JSONResponse(content={
        "has_reference": GOLDEN_SCHEMA.exists(),
        "ref_image_url": ref_image_url,
        "active_reference": active,
    })


# Mount outputs so the frontend can display the processed images
app.mount("/outputs", StaticFiles(directory=str(BASE_DIR.parent / "demo_output")), name="outputs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
