"""
WheelGuard API — FastAPI backend
Analyzes PyTorch models and replaces unstable layers with Wheel algebraic equivalents.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile, os, shutil
from analyzer import analyze_model

app = FastAPI(
    title="WheelGuard API",
    description="Algebraic NaN protection for PyTorch models via Wheel arithmetic",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "online", "service": "WheelGuard API", "version": "0.1.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Upload a PyTorch model (.pt / .pth) and get a full stability report.
    Returns detected NaN-prone layers and Wheel replacement recommendations.
    """
    if not file.filename.endswith(('.pt', '.pth')):
        raise HTTPException(status_code=400, detail="Only .pt and .pth files are supported.")

    # Save uploaded file to temp dir
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        report = analyze_model(tmp_path)
        return JSONResponse(content=report)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
