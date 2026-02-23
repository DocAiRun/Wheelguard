# WheelGuard

> Algebraic NaN protection for PyTorch models via Wheel arithmetic.

**WheelGuard** analyzes your PyTorch model, detects all NaN-prone layers (Softmax, LayerNorm, Attention), and replaces them with algebraically sound [WheelGrad](https://github.com/DocAiRun/wheelgrad) equivalents.

## Live Demo

🔗 [wheelguard.onrender.com](https://wheelguard-frontend.onrender.com)

## Architecture

```
wheelguard/
├── frontend/        ← Landing page (static HTML)
├── backend/         ← FastAPI API (Python)
│   ├── main.py      ← API routes
│   ├── analyzer.py  ← PyTorch layer scanner
│   ├── requirements.txt
│   └── Dockerfile
└── render.yaml      ← Render deployment config
```

## Run locally

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
# API running at http://localhost:8000
```

**Frontend:**
```bash
# Just open frontend/index.html in your browser
# or serve with:
python -m http.server 3000 --directory frontend
```

## API

```bash
# Analyze a model
curl -X POST http://localhost:8000/analyze \
  -F "file=@your_model.pt"

# Returns JSON report with all NaN-prone layers detected
```

## Powered by

- [WheelGrad](https://pypi.org/project/wheelgrad/) — Wheel algebra for PyTorch
- [FastAPI](https://fastapi.tiangolo.com/)
- [Render](https://render.com/)

## License

MIT — Johan Imboula 2026
