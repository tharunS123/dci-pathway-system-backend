from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
async def root():
    return JSONResponse({"message": "Hello from FastAPI on Vercel!"})

# Vercel's Python runtime will call the ASGI app; no special handler required in many cases
import os
import math
import pathlib
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dataclasses import asdict

# ------------------------------------------------------------
# Import your analyzer from root (because Vercel runs API from /api)
# ------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from sentimentAnalysis import analyze_csv   # your uploaded program

# ------------------------------------------------------------
# Directory Setup (Vercel serverless: /tmp is writable)
# ------------------------------------------------------------
MOM_DATA_DIR = os.path.join(ROOT, "data")         # CSVs stored in repo
STATIC_DIR = "/tmp/charts"                        # charts generated at runtime

os.makedirs(MOM_DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------
app = FastAPI(title="Sentiment Analysis API")

# Enable CORS (public)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Sentiment API running on Vercel!"}

# ------------------------------------------------------------
# ANALYZE ENDPOINT — identical design to your local version
# ------------------------------------------------------------
@app.get("/analyze/mom/{mom_id}")
async def analyze_mom(mom_id: str):
    """
    Example:
    GET /analyze/mom/Ashley_McCracklin
    Loads data/Ashley_McCracklin.csv
    Generates charts → /tmp (serverless)
    Returns cleaned JSON
    """

    csv_path = os.path.join(MOM_DATA_DIR, f"{mom_id}.csv")

    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=404,
            detail=f"CSV not found for mom_id '{mom_id}'. Expected file: {csv_path}",
        )

    try:
        # Run analyzer exactly like local version
        result = analyze_csv(
            csv_path=csv_path,
            mom_name=mom_id,
            static_dir=STATIC_DIR,
        )

        payload = asdict(result)

        # -----------------------------------
        # CLEAN JSON
        # -----------------------------------

        # Remove local PNG image references (since Vercel can't serve them)
        payload.pop("trend_img_url", None)
        payload.pop("themes_img_url", None)

        # tuples → lists
        if isinstance(payload.get("top_themes"), list):
            payload["top_themes"] = [list(x) for x in payload["top_themes"]]

        if isinstance(payload.get("trend_points"), list):
            payload["trend_points"] = [list(x) for x in payload["trend_points"]]

        # Replace NaN → None
        for k, v in payload.items():
            if isinstance(v, float) and math.isnan(v):
                payload[k] = None

        return payload

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze CSV: {str(e)}",
        )


# ------------------------------------------------------------
# Required for Vercel Python serverless
# ------------------------------------------------------------
# Vercel automatically calls the ASGI app, no run() needed.
