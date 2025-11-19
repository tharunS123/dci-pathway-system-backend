from flask import Flask, jsonify
from flask_cors import CORS
from dataclasses import asdict
import os

from analysis.sentimentAnalysis import analyze_csv

# Import your analysis module

app = Flask(__name__)
CORS(app)

# Path to all mom CSV files
MOM_DATA_DIR = "data"
STATIC_DIR = "static/charts"

os.makedirs(MOM_DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

@app.route("/")
def home():
    return jsonify({"status": "ok"}), 200

# ----------------------------
#   ROUTE: /analyze/mom/<id>
# ----------------------------
@app.route("/analyze/mom/<mom_id>", methods=["GET"])
def analyze_mom(mom_id):
    """
    Example:
    GET /analyze/mom/Ashley_McCracklin
    """

    # Build file path → mom_data/<id>.csv
    csv_path = os.path.join(MOM_DATA_DIR, f"{mom_id}.csv")

    # Validate file exists
    if not os.path.exists(csv_path):
        return jsonify({
            "error": f"CSV not found for mom_id '{mom_id}'. Expected file: {csv_path}"
        }), 404

    try:
        # Run your ML-powered analysis
        result = analyze_csv(
            csv_path=csv_path,
            mom_name=mom_id,
            static_dir=STATIC_DIR
        )

        payload = asdict(result)

        # Remove image-related fields since you don't need them in the API
        payload.pop("trend_img_url", None)
        payload.pop("themes_img_url", None)

        # Convert dataclass → JSON
        return jsonify(asdict(result)), 200

    except Exception as e:
        return jsonify({
            "error": "Failed to analyze CSV",
            "details": str(e)
        }), 500


# ----------------------------
#   HEALTH CHECK (Optional)
# ----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ----------------------------
#   RUN FLASK APP
# ----------------------------
if __name__ == "__main__":
    # Important: host=0.0.0.0 if deploying (Vercel / Render / EC2)
    app.run(host="0.0.0.0", port=5000, debug=True)
