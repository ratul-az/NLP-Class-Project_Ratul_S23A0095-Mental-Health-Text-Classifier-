from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your saved pipeline (TF-IDF + LogisticRegression)
# Make sure the path matches your folder structure
MODEL_PATH = "artifacts/baseline_tfidf_logreg.joblib"
model = joblib.load(MODEL_PATH)


def label_theme(label: str) -> dict:
    """
    Map labels to UI themes (colors).
    Adjust these mappings to match your dataset labels.
    """
    key = (label or "").strip().lower()

    # Default theme
    theme = {
        "name": "default",
        "accent": "#7c3aed",   # purple
        "bg": "rgba(124, 58, 237, 0.12)",
        "border": "rgba(124, 58, 237, 0.35)"
    }

    # Common label-based themes (edit as you like)
    if "anx" in key or "anxiety" in key:
        theme.update({"name": "anxiety", "accent": "#f59e0b", "bg": "rgba(245, 158, 11, 0.12)", "border": "rgba(245, 158, 11, 0.35)"})
    elif "dep" in key or "depression" in key:
        theme.update({"name": "depression", "accent": "#3b82f6", "bg": "rgba(59, 130, 246, 0.12)", "border": "rgba(59, 130, 246, 0.35)"})
    elif "stress" in key:
        theme.update({"name": "stress", "accent": "#ef4444", "bg": "rgba(239, 68, 68, 0.12)", "border": "rgba(239, 68, 68, 0.35)"})
    elif "suic" in key or "self" in key:
        theme.update({"name": "critical", "accent": "#fb7185", "bg": "rgba(251, 113, 133, 0.12)", "border": "rgba(251, 113, 133, 0.35)"})
    elif "bip" in key or "mania" in key:
        theme.update({"name": "bipolar", "accent": "#22c55e", "bg": "rgba(34, 197, 94, 0.12)", "border": "rgba(34, 197, 94, 0.35)"})
    elif "normal" in key or "none" in key or "healthy" in key:
        theme.update({"name": "normal", "accent": "#10b981", "bg": "rgba(16, 185, 129, 0.12)", "border": "rgba(16, 185, 129, 0.35)"})

    return theme


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "").strip()
    if not text:
        return render_template(
            "index.html",
            text="",
            prediction=None,
            confidence=None,
            confidence_pct=0,
            theme=None,
            topk=None
        )

    probs = model.predict_proba([text])[0]
    idx = int(np.argmax(probs))
    pred_label = str(model.classes_[idx])
    conf = float(probs[idx])
    conf_pct = int(round(conf * 100))

    # Top-3 (optional display)
    k = min(3, len(model.classes_))
    top_idx = np.argsort(probs)[::-1][:k]
    topk = [
        {"label": str(model.classes_[i]), "pct": int(round(float(probs[i]) * 100))}
        for i in top_idx
    ]

    theme = label_theme(pred_label)

    return render_template(
        "index.html",
        text=text,
        prediction=pred_label,
        confidence=round(conf, 4),
        confidence_pct=conf_pct,
        theme=theme,
        topk=topk
    )


if __name__ == "__main__":
    app.run(debug=True)
