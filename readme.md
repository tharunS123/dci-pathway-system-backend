# 📊 Dream Center Indy Backend

This repository contains a **FastAPI-based backend** for running analysis on MOM (Manager on Duty) CSV files.  
The API processes uploaded or stored `.csv` files, performs NLP-like keyword sentiment scoring, generates weekly trend charts, extracts themes, and returns a structured JSON payload.

This backend is designed to run on **Vercel Serverless Functions**, with charts generated in a temporary runtime directory (`/tmp`), which is the only writable path on Vercel Lambdas.

---

## 🚀 Features

- ✔️ FastAPI serverless backend  
- ✔️ CORS enabled (for frontend integration)  
- ✔️ Analyze MOM CSV files using `sentimentAnalysis.py`  
- ✔️ Weekly sentiment trend calculation  
- ✔️ Theme extraction and scoring  
- ✔️ Auto-clean JSON responses (e.g., NaN → `null`)  
- ✔️ Chart generation using Matplotlib (saved to `/tmp/charts`)  
- ✔️ Compatible with Vercel Python serverless runtime  
- ✔️ Supports endpoint:
  - `GET /analyze/mom/{mom_id}`  
- ✔️ Easy React or Next.js frontend integration  

---

## 📁 Repository Structure


**Note:**  
- `data/` holds your MOM CSV files.  
- Vercel only allows writing to `/tmp` → charts are generated there.

---

## 🧪 Local Development

### 1️⃣ Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run FastAPI locally
```bash
uvicorn api.index:app --reload
```
### 4️⃣ Test endpoint Example
```bash
curl http://localhost:8000/analyze/mom/Ashley_McCracklin
```

---

## 📡 API Endpoints

### GET / Health check.
```json
{
  "status": "ok",
  "message": "Sentiment API running on Vercel!"
}
```

### GET /analyze/mom/{mom_id}
Analyze a CSV with the filename:
```kotlin
data/{mom_id}.csv
```

#### Example Request
```pgsql
GET /analyze/mom/Ashley_McCracklin
```

#### Example CSV Path
```kotlin
data/Ashley_McCracklin.csv
```

#### Example Response (trimmed)
```json
{
  "mom_name": "Ashley_McCracklin",
  "rows": 1488,
  "pos_total": 225,
  "neg_total": 118,
  "trend_points": [
    ["2024-W01", 12],
    ["2024-W02", -3]
  ],
  "top_themes": [
    ["communication", 14],
    ["training", 9]
  ],
  "preview_html": "<table>...</table>"
}
```

---

## 🏗 Deployment (Vercel)

### 1️⃣ Install Vercel CLI
```bash
npm i -g vercel
```

### 2️⃣ Deploy
```bash
vercel --prod
```

### 3️⃣ Required file: vercel.json
```json
{
  "version": 2,
  "routes": [
    { "src": "/(.*)", "dest": "/api/index.py" }
  ]
}
```

--- 

## 📦 Working With CSV Data

#### Place all MOM CSV files inside:
```kotlin
data/{mom_id}.csv
```

#### Example:
```kotlin
data/Ashley_McCracklin.csv
data/Michael_Lopez.csv
```
#### Each name matches the URL parameter for the endpoint.

---

## 📄 License
MIT License — free for personal and commercial use.