import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Keyword lists (extend as needed) ---
POSITIVE = {
    "job","hired","interview","offer","promotion","raise","passed","improved",
    "on track","progress","completed","achieved","approved","secured","saved",
    "budget","stable","positive","better","increased","support","helped",
    "enrolled","certified","degree","class","training","childcare","housing",
    "transportation","health","therapy","resolved","win","winning","success"
}
NEGATIVE = {
    "late","missed","declined","failed","fired","lost","no show","negative",
    "worse","decrease","issue","problem","barrier","delay","denied","crisis",
    "eviction","homeless","ill","sick","depressed","anxious","stressed",
    "lack","shortage","overdue","debt","fine","ticket","conflict","absent"
}

# Weighted sentiment dictionaries tuned for DCI-style notes.
# These capture that some milestones or crises matter more than simple word counts.
POSITIVE_WEIGHTS = {
    # Employment milestones
    "hired": 5,"job offer": 4,"offer": 3,"interview": 2,"passed interview": 4,"employment": 2,"promotion": 4,"new job": 5,
    "secured income": 4,"raise": 3,"budget completed": 3,"saved": 2,"on track financially": 3, # Financial stability
    "secured housing": 5,"approved": 4,"rent support": 3,"landlord agreement": 3,"utilities stable": 2, # Housing stability
    "license approved": 4,"insurance updated": 2,"got car": 5,"transport secured": 3,"ride support": 1, # Transportation
    "enrolled": 3,"completed training": 4,"certified": 4,"graduated": 5,"progress": 1, # Education / training
    "secured childcare": 5,"daycare approved": 4, # Childcare
    "therapy progress": 2,"medication stable": 2,"health improving": 3,"mental health positive": 3, # Health and mental health
    "achieved": 3,"improved": 2,"success": 3,"resolved": 2,"support received": 2, # Generic strong positives
}

NEGATIVE_WEIGHTS = {
    # Employment issues
    "fired": 5,
    "lost job": 5,
    "missed interview": 3,

    # Housing crises
    "eviction": 5,
    "homeless": 5,
    "late rent": 3,
    "housing issue": 2,

    # Financial barriers
    "debt": 3,
    "overdue": 2,
    "missed payment": 3,
    "fine": 2,

    # Transportation problems
    "no transportation": 3,
    "car broke": 3,
    "license suspended": 4,

    # Childcare barriers
    "no childcare": 3,
    "childcare unavailable": 2,

    # Health crises
    "ill": 2,
    "sick": 1,
    "mental health crisis": 4,
    "depressed": 4,
    "anxious": 2,
    "stressed": 2,

    # Legal issues
    "court": 2,
    "probation issue": 3,
    "ticket": 1,

    # Generic negatives
    "problem": 1,
    "barrier": 2,
    "declined": 2,
    "missed": 1,
    "no show": 2,
}


THEMES = {
    "employment": ["job","work","interview","resume","offer","hire","schedule","shift","promotion"],
    "education":  ["school","class","course","degree","ged","certificate","enroll","training","study"],
    "finance":    ["budget","saved","saving","debt","bill","income","pay","raise","fine","tax"],
    "housing":    ["housing","rent","lease","landlord","eviction","homeless","shelter","apartment"],
    "transport":  ["bus","car","ride","uber","lyft","license","insurance","transport","commute"],
    "childcare":  ["childcare","daycare","babysit","pre-k","kindergarten","after-school"],
    "health":     ["health","doctor","clinic","therapy","counsel","mental","medication","sick","ill"],
    "legal":      ["court","case","probation","ticket","fine","legal","attorney","custody"],
}

@dataclass
class AnalysisResult:
    mom_name: str
    rows: int
    start: Optional[str]
    end: Optional[str]
    pos_total: int
    neg_total: int
    net_total: int
    pos_rate: float
    top_themes: List[Tuple[str,int]]
    trend_img_url: Optional[str]
    themes_img_url: Optional[str]
    preview_html: str
    trend_dir: str
    best_week_label: Optional[str]
    best_week_value: Optional[float]
    worst_week_label: Optional[str]
    worst_week_value: Optional[float]
    trend_points: List[Tuple[str, float]]

# ---------- helpers ----------

def _infer_date_column(df: pd.DataFrame) -> Optional[str]:
    prefer = ["date","week","week_date","created","timestamp","time","entry_date"]
    cols_low = {c.lower(): c for c in df.columns}
    for p in prefer:
        if p in cols_low:
            return cols_low[p]
    # fallback: anything containing "date" or "week" that parses
    for c in df.columns:
        cl = c.lower()
        if "date" in cl or "week" in cl:
            try:
                pd.to_datetime(df[c], errors="raise")
                return c
            except Exception:
                continue
    return None

_WORD = re.compile(r"[a-zA-Z']+")

def _row_text(df: pd.DataFrame) -> pd.Series:
    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not text_cols:
        return pd.Series([""] * len(df), index=df.index)
    return df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

def _count_keywords(text: str, vocab: set) -> int:
    if not text or not isinstance(text, str):
        return 0
    t = text.lower()
    # multi-word first
    multi = [w for w in vocab if " " in w]
    count = 0
    for m in multi:
        count += t.count(m)
    words = _WORD.findall(t)
    count += sum(1 for w in words if w in vocab)
    return count

def _count_weighted(text: str, weights: dict) -> int:
    """Return a weighted sentiment score for a given piece of text.

    `weights` is a mapping from word/phrase -> positive weight (for positives)
    or magnitude (for negatives). Multi-word phrases are matched on substrings
    first, then single tokens are matched using the _WORD regex.
    """
    if not text or not isinstance(text, str):
        return 0
    t = text.lower()
    score = 0

    # multi-word phrases first
    for phrase, w in weights.items():
        if " " in phrase and phrase in t:
            score += w

    # then token-level matches for single words
    tokens = _WORD.findall(t)
    for tok in tokens:
        if tok in weights and " " not in tok:
            score += weights[tok]

    return score

def _themes_for_text(t: str):
    t_low = (t or "").lower()
    for th, words in THEMES.items():
        for w in words:
            if w in t_low:
                yield th
                break

def _safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")


# ---------- main analysis ----------

def analyze_csv(csv_path: str, mom_name: Optional[str], static_dir: str) -> AnalysisResult:
    df = pd.read_csv(csv_path)
    text = _row_text(df)

    # Weighted sentiment using DCI-tuned keyword weights.
    # pos = sum of positive weights; neg = sum of negative weights (as a positive magnitude).
    pos = text.apply(lambda s: _count_weighted(s, POSITIVE_WEIGHTS))
    neg = text.apply(lambda s: _count_weighted(s, NEGATIVE_WEIGHTS))
    net = pos - neg

    date_col = _infer_date_column(df)
    if date_col:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        df["_dt"] = dt
        week = dt.dt.to_period("W").astype(str)
        start = str(dt.min().date()) if dt.notna().any() else None
        end   = str(dt.max().date()) if dt.notna().any() else None
    else:
        df["_dt"] = pd.NaT
        week = pd.Series([None] * len(df))
        start = end = None

    # trend by week
    trend = pd.DataFrame({"week": week, "net": net}).groupby("week", dropna=False)["net"].mean()
    best_week_label = worst_week_label = None
    best_week_value = worst_week_value = None
    if len(trend) > 0:
        best_week_value = float(trend.max())
        best_week_label = str(trend.idxmax())
        worst_week_value = float(trend.min())
        worst_week_label = str(trend.idxmin())

    # convert trend Series into a list of (week, value) pairs
    trend_points = [(str(idx), float(val)) for idx, val in trend.items()]

    if len(trend) >= 3:
        y = trend.tail(min(6, len(trend)))
        x = np.arange(len(y))
        slope = np.polyfit(x, y.values.astype(float), 1)[0]
        if slope > 0.05:
            trend_dir = "up"
        elif slope < -0.05:
            trend_dir = "down"
        else:
            trend_dir = "flat"
    else:
        trend_dir = "flat"

    # themes
    theme_counts = {k: 0 for k in THEMES.keys()}
    for s in text.fillna(""):
        for th in _themes_for_text(s):
            theme_counts[th] += 1
    top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)

    # images
    os.makedirs(static_dir, exist_ok=True)
    base = _safe_filename(mom_name or os.path.splitext(os.path.basename(csv_path))[0])

    # trend line
    trend_img = None
    if len(trend) > 0:
        plt.figure(figsize=(6,3.2))
        trend.plot(marker="o")
        plt.title(f"Weekly Sentiment Trend — {mom_name or base}")
        plt.xlabel("Week"); plt.ylabel("Mean weighted sentiment (pos - neg)")
        plt.tight_layout()
        fname = f"trend_{base}.png"
        plt.savefig(os.path.join(static_dir, fname))
        plt.close()
        trend_img = f"/charts/{fname}"

    # theme bar
    themes_img = None
    names = [k for k, v in top_themes[:8] if v > 0]
    vals  = [v for k, v in top_themes[:8] if v > 0]
    if names and vals:
        plt.figure(figsize=(5,3))
        plt.bar(names, vals)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Rows mentioning theme")
        plt.title(f"Key Themes — {mom_name or base}")
        plt.tight_layout()
        fname2 = f"themes_{base}.png"
        plt.savefig(os.path.join(static_dir, fname2))
        plt.close()
        themes_img = f"/charts/{fname2}"

    # preview table
    preview_cols = []
    for c in df.columns:
        if c == "_dt":
            continue
        if len(preview_cols) >= 6:
            break
        preview_cols.append(c)
    preview = df[preview_cols].head(10).to_html(
        classes="table table-striped table-sm preview",
        index=False,
        border=0
    )

    pos_total = int(pos.sum())
    neg_total = int(neg.sum())
    net_total = int(net.sum())
    pos_rate  = float((pos > 0).mean()) if len(pos) else 0.0

    return AnalysisResult(
        mom_name=mom_name or base,
        rows=len(df),
        start=start, end=end,
        pos_total=pos_total, neg_total=neg_total, net_total=net_total,
        pos_rate=pos_rate,
        top_themes=top_themes[:5],
        trend_img_url=trend_img,
        themes_img_url=themes_img,
        preview_html=preview,
        trend_dir=trend_dir,
        best_week_label=best_week_label, best_week_value=best_week_value,
        worst_week_label=worst_week_label, worst_week_value=worst_week_value,
        trend_points=trend_points
    )
