"""Aplikasi Flask: prediksi churn pelanggan dengan Decision Tree (TA-05)."""
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "churn_decision_tree.joblib"

app = Flask(__name__)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model tidak ditemukan di {MODEL_PATH}. Jalankan notebook TA05 sampai bagian 7 lebih dulu."
    )

model = joblib.load(MODEL_PATH)
prep = model.named_steps["prep"]
clf = model.named_steps["clf"]

NUM_COLS = list(prep.transformers_[0][2])
CAT_COLS = list(prep.transformers_[1][2])
CATEGORIES = dict(zip(CAT_COLS, prep.named_transformers_["cat"].categories_))
FEATURE_NAMES = list(prep.get_feature_names_out())

# ---------- Label tampilan (nilai yang dikirim ke model tetap nilai asli dataset) ----------
FIELD_LABELS = {
    "tenure": "Lama berlangganan (bulan)",
    "MonthlyCharges": "Tagihan bulanan (USD)",
    "TotalCharges": "Total tagihan sejauh ini (USD)",
    "Contract": "Jenis kontrak",
    "InternetService": "Layanan internet",
    "OnlineSecurity": "Online Security",
    "TechSupport": "Tech Support",
    "PaymentMethod": "Metode pembayaran",
    "PaperlessBilling": "Tagihan tanpa kertas",
}
OPTION_LABELS = {
    "Month-to-month": "Bulanan (month-to-month)",
    "One year": "1 tahun",
    "Two year": "2 tahun",
    "Yes": "Ya",
    "No": "Tidak",
    "No internet service": "Tanpa layanan internet",
    "Electronic check": "Cek elektronik",
    "Mailed check": "Cek lewat pos",
    "Bank transfer (automatic)": "Transfer bank (otomatis)",
    "Credit card (automatic)": "Kartu kredit (otomatis)",
}
FIELD_GROUPS = [
    ("Akun", ["tenure", "Contract", "PaymentMethod", "PaperlessBilling"]),
    ("Layanan", ["InternetService", "OnlineSecurity", "TechSupport"]),
    ("Tagihan", ["MonthlyCharges", "TotalCharges"]),
]
NUM_LIMITS = {"tenure": (0, 120), "MonthlyCharges": (0, 500), "TotalCharges": (0, 100000)}
DEFAULTS = {"tenure": "12", "MonthlyCharges": "70", "TotalCharges": ""}


def option_label(field, value):
    if field == "InternetService" and value == "No":
        return "Tidak berlangganan internet"
    return OPTION_LABELS.get(value, value)


def field_options(field):
    return [(v, option_label(field, v)) for v in CATEGORIES[field]]


def default_values():
    values = dict(DEFAULTS)
    for col in CAT_COLS:
        values[col] = CATEGORIES[col][0]
    return values


def parse_form(form):
    """Validasi input. Mengembalikan (baris_data, error)."""
    errors, row = [], {}
    for col in ["tenure", "MonthlyCharges"]:
        raw = (form.get(col) or "").strip().replace(",", ".")
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{FIELD_LABELS[col]} harus berupa angka.")
            continue
        lo, hi = NUM_LIMITS[col]
        if not lo <= val <= hi:
            errors.append(f"{FIELD_LABELS[col]} harus antara {lo} dan {hi}.")
            continue
        row[col] = val

    raw_total = (form.get("TotalCharges") or "").strip().replace(",", ".")
    if raw_total:
        try:
            total = float(raw_total)
            if not 0 <= total <= NUM_LIMITS["TotalCharges"][1]:
                raise ValueError
            row["TotalCharges"] = total
        except ValueError:
            errors.append(f"{FIELD_LABELS['TotalCharges']} harus berupa angka positif, atau kosongkan.")
    elif "tenure" in row and "MonthlyCharges" in row:
        # Dikosongkan -> perkiraan: lama berlangganan x tagihan bulanan
        row["TotalCharges"] = round(row["tenure"] * row["MonthlyCharges"], 2)

    for col in CAT_COLS:
        val = form.get(col)
        if val not in CATEGORIES[col]:
            errors.append(f"Pilihan untuk {FIELD_LABELS[col]} tidak valid.")
        else:
            row[col] = val
    return row, errors


def explain_path(X):
    """Terjemahkan jalur keputusan pohon untuk satu pelanggan menjadi daftar pertanyaan-jawaban."""
    Xt = prep.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    tree = clf.tree_
    steps = []
    for node in clf.decision_path(Xt).indices:
        if tree.children_left[node] == tree.children_right[node]:  # daun
            leaf = node
            continue
        name = FEATURE_NAMES[tree.feature[node]]
        value, thr = Xt[0, tree.feature[node]], tree.threshold[node]
        kind, rest = name.split("__", 1)
        if kind == "num":
            question = f"{FIELD_LABELS[rest]} ≤ {thr:.1f}?"
            note = f"nilai pelanggan: {value:g}"
            answer = "Ya" if value <= thr else "Tidak"
        else:
            col, cat = rest.split("_", 1)
            question = f"{FIELD_LABELS[col]} = {option_label(col, cat)}?"
            note = None
            answer = "Ya" if value > thr else "Tidak"
        steps.append({"question": question, "answer": answer, "note": note})
    return steps, leaf


def predict(row):
    X = pd.DataFrame([row], columns=NUM_COLS + CAT_COLS)
    proba = float(model.predict_proba(X)[0, list(clf.classes_).index(1)])
    pred = int(model.predict(X)[0])
    steps, leaf = explain_path(X)
    return {
        "churn": pred == 1,
        "label": "Berisiko churn" if pred == 1 else "Cenderung bertahan",
        "pct": round(proba * 100, 1),
        "bars": max(0, min(5, round((1 - proba) * 5))),
        "steps": steps,
        "leaf_samples": int(clf.tree_.n_node_samples[leaf]),
    }


def render(values, result=None, errors=None):
    return render_template(
        "index.html",
        groups=FIELD_GROUPS,
        labels=FIELD_LABELS,
        options=field_options,
        num_fields=NUM_COLS,
        values=values,
        result=result,
        errors=errors or [],
    )


@app.get("/")
def index():
    return render(default_values())


@app.post("/predict")
def predict_view():
    values = default_values()
    values.update({k: v for k, v in request.form.items() if k in values})
    row, errors = parse_form(request.form)
    if errors:
        return render(values, errors=errors), 400
    return render(values, result=predict(row))


@app.get("/predict")
def predict_redirect():
    return redirect(url_for("index"))


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True)
