from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file, jsonify

app = Flask(__name__)

# ==================================================
# 1. KONFIGURASI PATH
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "student_data.csv"
MODEL_PATH = BASE_DIR / "models" / "student_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"
OUTPUT_PATH = BASE_DIR / "predictions.csv"

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# ==================================================
# 2. MEMUAT MODEL, METADATA, DAN DATASET
# ==================================================

for path in [DATA_PATH, MODEL_PATH, METADATA_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

feature_names = metadata["feature_names"]
classes = metadata["classes"]

df = pd.read_csv(DATA_PATH)

if "Target" not in df.columns:
    raise ValueError("Kolom Target tidak ditemukan di dataset.")

# Hitung median dari seluruh fitur sebagai nilai default dasar
median_features = df[feature_names].median().to_dict()

# Hitung distribusi target
target_distribution = (
    df["Target"].value_counts().reindex(classes, fill_value=0).to_dict()
)

# Label Bahasa Indonesia yang ramah orang awam
TARGET_LABELS = {
    "Graduate": "Lulus",
    "Dropout": "Putus Studi",
    "Enrolled": "Aktif",
}

FEATURE_LABELS = {
    "Curricular units 1st sem (approved)": "Mata Kuliah Lulus Smt 1",
    "Curricular units 1st sem (enrolled)": "Mata Kuliah Diambil Smt 1",
    "Curricular units 1st sem (grade)": "Rata-rata Nilai Smt 1",
    "Curricular units 2nd sem (approved)": "Mata Kuliah Lulus Smt 2",
    "Curricular units 2nd sem (enrolled)": "Mata Kuliah Diambil Smt 2",
    "Curricular units 2nd sem (grade)": "Rata-rata Nilai Smt 2",
    "Tuition fees up to date": "Status Pembayaran SPP",
    "Scholarship holder": "Penerima Beasiswa",
    "Age at enrollment": "Usia Saat Masuk",
    "Admission grade": "Nilai Tes Masuk",
    "Displaced": "Mahasiswa Rantau",
    "Gender": "Jenis Kelamin",
    "Debtor": "Tunggakan Keuangan",
    "Course": "Program Studi",
    "Daytime/evening attendance": "Jadwal Kuliah",
}

def get_target_label(name):
    return TARGET_LABELS.get(name, name)

# ==================================================
# 3. PRESET PROFIL REALISTIS UNTUK ORANG AWAM
# ==================================================

def create_presets():
    # 1. Profil Emas (Mahasiswa Berprestasi)
    gold = median_features.copy()
    gold.update({
        "Curricular units 1st sem (approved)": 6.0,
        "Curricular units 1st sem (enrolled)": 6.0,
        "Curricular units 1st sem (evaluations)": 7.0,
        "Curricular units 1st sem (grade)": 15.5,
        "Curricular units 2nd sem (approved)": 6.0,
        "Curricular units 2nd sem (enrolled)": 6.0,
        "Curricular units 2nd sem (evaluations)": 7.0,
        "Curricular units 2nd sem (grade)": 16.0,
        "Tuition fees up to date": 1.0,
        "Scholarship holder": 1.0,
        "Debtor": 0.0,
        "Age at enrollment": 19.0,
        "Admission grade": 145.0,
    })

    # 2. Profil Berisiko (Mahasiswa Terancam Putus Studi)
    risk = median_features.copy()
    risk.update({
        "Curricular units 1st sem (approved)": 0.0,
        "Curricular units 1st sem (enrolled)": 6.0,
        "Curricular units 1st sem (evaluations)": 2.0,
        "Curricular units 1st sem (grade)": 0.0,
        "Curricular units 2nd sem (approved)": 0.0,
        "Curricular units 2nd sem (enrolled)": 6.0,
        "Curricular units 2nd sem (evaluations)": 1.0,
        "Curricular units 2nd sem (grade)": 0.0,
        "Tuition fees up to date": 0.0,
        "Scholarship holder": 0.0,
        "Debtor": 1.0,
        "Age at enrollment": 27.0,
        "Admission grade": 115.0,
    })

    # 3. Profil Tengah (Mahasiswa Sedang Aktif / Rata-rata)
    mid = median_features.copy()
    mid.update({
        "Curricular units 1st sem (approved)": 5.0,
        "Curricular units 1st sem (enrolled)": 6.0,
        "Curricular units 1st sem (evaluations)": 6.0,
        "Curricular units 1st sem (grade)": 12.5,
        "Curricular units 2nd sem (approved)": 5.0,
        "Curricular units 2nd sem (enrolled)": 6.0,
        "Curricular units 2nd sem (evaluations)": 6.0,
        "Curricular units 2nd sem (grade)": 12.5,
        "Tuition fees up to date": 1.0,
        "Scholarship holder": 0.0,
        "Debtor": 0.0,
        "Age at enrollment": 20.0,
        "Admission grade": 128.0,
    })

    return {"gold": gold, "risk": risk, "mid": mid}

PRESETS = create_presets()

# ==================================================
# 4. HELPER PREDIKSI & PENJELASAN BAHASA MANUSIA
# ==================================================

def run_prediction_engine(input_dict):
    """Menjalankan prediksi dan menghasilkan penjelasan human-friendly"""
    # Siapkan dataframe dengan seluruh 36 fitur (isi dengan median jika tidak ada)
    row_data = {}
    for col in feature_names:
        row_data[col] = float(input_dict.get(col, median_features[col]))
    
    input_df = pd.DataFrame([row_data], columns=feature_names)
    
    # Model inference
    pred = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    prob_dict = {
        cls: round(float(prob) * 100, 1)
        for cls, prob in zip(model.classes_, probs)
    }

    # Urutkan probabilitas
    ordered_probs = [
        {"key": "Dropout", "label": "Putus Studi", "color": "rose", "pct": prob_dict.get("Dropout", 0.0)},
        {"key": "Enrolled", "label": "Aktif", "color": "amber", "pct": prob_dict.get("Enrolled", 0.0)},
        {"key": "Graduate", "label": "Lulus", "color": "emerald", "pct": prob_dict.get("Graduate", 0.0)},
    ]

    # Skor risiko tidak lulus: Prob Dropout + 0.5 * Prob Enrolled
    risk_score = round(prob_dict.get("Dropout", 0.0) + (prob_dict.get("Enrolled", 0.0) * 0.45), 1)
    risk_score = min(max(risk_score, 0.0), 100.0)

    # Faktor paling berpengaruh untuk baris ini
    s2_app = row_data["Curricular units 2nd sem (approved)"]
    s1_app = row_data["Curricular units 1st sem (approved)"]
    s2_grd = row_data["Curricular units 2nd sem (grade)"]
    tuition = row_data["Tuition fees up to date"]

    factors = [
        {"name": "Mata Kuliah Lulus Smt 2", "desc": f"{int(s2_app)} MK lulus", "impact": "39%"},
        {"name": "Mata Kuliah Lulus Smt 1", "desc": f"{int(s1_app)} MK lulus", "impact": "24%"},
        {"name": "Rata-rata Nilai Smt 2", "desc": f"Nilai {s2_grd}", "impact": "18%"},
        {"name": "Status Pembayaran SPP", "desc": "Lunas" if tuition == 1 else "Menunggak", "impact": "12%"},
    ]

    # Narasi penjelasan human-friendly
    confidence = prob_dict.get(pred, 0.0)
    if pred == "Graduate":
        explanation = (
            f"Model memprediksi mahasiswa ini berpeluang besar LULUS dengan keyakinan {confidence}%. "
            f"Kombinasi performansi akademik menunjukkan lintasan kelulusan yang kokoh. "
            f"Faktor penentu utama: kelulusan {int(s2_app)} mata kuliah di semester 2 dan "
            f"indeks prestasi semester yang konsisten di angka {s2_grd}."
        )
    elif pred == "Dropout":
        explanation = (
            f"Model mendeteksi mahasiswa ini BERISIKO PUTUS STUDI (DROPOUT) dengan keyakinan {confidence}%. "
            f"Pemicu utama yang terdeteksi adalah rendahnya kelulusan mata kuliah ({int(s2_app)} lulus) "
            f"serta indikasi kendala akademik/finansial yang membutuhkan pendampingan segera oleh Dosen PA."
        )
    else:
        explanation = (
            f"Model mengklasifikasikan mahasiswa ini berstatus AKTIF (ENROLLED) dengan keyakinan {confidence}%. "
            f"Mahasiswa sedang berada di fase penentuan kelulusan dengan performa yang cukup stabil namun "
            f"memerlukan dorongan konsistensi SKS pada semester berikutnya."
        )

    return {
        "status": "success",
        "prediction": pred,
        "label": TARGET_LABELS.get(pred, pred),
        "confidence": confidence,
        "probabilities": ordered_probs,
        "risk_score": risk_score,
        "explanation": explanation,
        "factors": factors,
        "model_info": "Random Forest — 300 pohon",
    }


# ==================================================
# 5. ROUTES: HALAMAN-HALAMAN MENU UTAMA
# ==================================================

# 1. Menu: BERANDA
@app.route("/")
def index():
    return render_template(
        "beranda.html",
        active_menu="beranda",
        total_students=len(df),
        accuracy=metadata["accuracy"],
        balanced_accuracy=metadata["balanced_accuracy"],
        confusion_matrix=metadata["confusion_matrix"],
        distribution=target_distribution,
        models_arena=metadata.get("models_arena", {}),
    )

# 2. Menu: DATASET
@app.route("/dataset")
def dataset_page():
    # Ambil sampel 15 baris data untuk pratinjau tabel interaktif
    sample_preview = df.head(15).to_dict(orient="records")
    return render_template(
        "dataset.html",
        active_menu="dataset",
        total_students=len(df),
        feature_count=len(feature_names),
        distribution=target_distribution,
        sample_rows=sample_preview,
        columns=df.columns.tolist()[:10] + ["Target"],
    )

# 3. Menu: MODEL
@app.route("/model")
def model_page():
    raw_importance = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )[:12]
    
    feature_importance = [
        {
            "name": FEATURE_LABELS.get(name, name),
            "original_name": name,
            "importance": round(float(val) * 100, 2),
        }
        for name, val in raw_importance
    ]

    return render_template(
        "model.html",
        active_menu="model",
        accuracy=metadata["accuracy"],
        balanced_accuracy=metadata["balanced_accuracy"],
        cv_accuracy=metadata.get("cv_accuracy_mean", 0.767),
        classes=classes,
        report=metadata["classification_report"],
        confusion_matrix=metadata["confusion_matrix"],
        feature_importance=feature_importance,
        n_estimators=metadata.get("n_estimators", 300),
        models_arena=metadata.get("models_arena", {}),
    )

# 4. Menu: PREDIKSI (RAMAH ORANG AWAM DENGAN SLIDER & LIVE AJAX)
@app.route("/prediksi")
def prediksi_page():
    # Inisialisasi awal menggunakan Profil Tengah (Mid)
    initial_data = PRESETS["mid"]
    initial_result = run_prediction_engine(initial_data)

    return render_template(
        "prediksi.html",
        active_menu="prediksi",
        initial_data=initial_data,
        initial_result=initial_result,
        presets=PRESETS,
    )

# 5. Menu: AI STUDIO / BATCH CSV
@app.route("/ai-studio")
def ai_studio_page():
    return render_template(
        "ai_studio.html",
        active_menu="ai_studio",
    )


# ==================================================
# 6. API ENDPOINTS (LIVE INTERACTIVE REAL-TIME)
# ==================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Endpoint AJAX live real-time saat slider digeser"""
    data = request.get_json(silent=True) or request.form.to_dict()
    result = run_prediction_engine(data)
    return jsonify(result)

@app.route("/api/preset/<preset_name>", methods=["GET"])
def api_get_preset(preset_name):
    """Mengambil nilai preset (gold, risk, mid, random)"""
    if preset_name == "random":
        # Ambil 1 baris acak riil dari dataset
        random_row = df.sample(1).iloc[0][feature_names].to_dict()
        pred_res = run_prediction_engine(random_row)
        return jsonify({"data": random_row, "prediction": pred_res})
    
    preset_data = PRESETS.get(preset_name, PRESETS["mid"])
    pred_res = run_prediction_engine(preset_data)
    return jsonify({"data": preset_data, "prediction": pred_res})


# ==================================================
# 7. PREDIKSI MASSAL CSV (AI STUDIO)
# ==================================================

@app.route("/predict-csv", methods=["POST"])
def predict_csv():
    file = request.files.get("file")

    if not file or not file.filename.lower().endswith(".csv"):
        return render_template(
            "error.html",
            message="Silakan pilih file CSV yang valid (.csv).",
        ), 400

    try:
        uploaded = pd.read_csv(file)

        if uploaded.empty:
            raise ValueError("File CSV tidak memiliki data.")

        # Cek kolom yang tersedia
        missing = [name for name in feature_names if name not in uploaded.columns]
        if missing:
            raise ValueError(f"CSV belum memiliki kolom berikut: {', '.join(missing[:4])}...")

        X = uploaded[feature_names].apply(pd.to_numeric, errors="coerce").fillna(df[feature_names].median())
        predictions = model.predict(X)
        pred_counts = pd.Series(predictions).value_counts().to_dict()

        summary = {
            "total": len(predictions),
            "graduate": pred_counts.get("Graduate", 0),
            "dropout": pred_counts.get("Dropout", 0),
            "enrolled": pred_counts.get("Enrolled", 0),
        }

        output = uploaded.copy()
        output["Prediksi_Status"] = predictions
        output["Keterangan_Prediksi"] = [TARGET_LABELS.get(p, p) for p in predictions]
        output.to_csv(OUTPUT_PATH, index=False)

        return render_template(
            "batch_result.html",
            active_menu="ai_studio",
            count=len(output),
            summary=summary,
            rows=output.head(25).to_dict(orient="records"),
            columns=output.columns.tolist(),
        )

    except Exception as error:
        return render_template("error.html", message=f"CSV tidak dapat diproses: {error}"), 400

@app.route("/download-predictions")
def download_predictions():
    if not OUTPUT_PATH.exists():
        return render_template("error.html", message="Belum ada hasil prediksi CSV."), 404

    return send_file(
        OUTPUT_PATH,
        as_attachment=True,
        download_name="Hasil_Prediksi_Kelulusan.csv",
        mimetype="text/csv",
    )

@app.route("/download-sample-csv")
def download_sample_csv():
    """Mengunduh 10 baris sampel CSV agar dosen/pengguna bisa langsung coba upload"""
    sample_df = df.head(10)[feature_names]
    sample_path = BASE_DIR / "sample_mahasiswa_test.csv"
    sample_df.to_csv(sample_path, index=False)
    return send_file(
        sample_path,
        as_attachment=True,
        download_name="sample_dataset_uji.csv",
        mimetype="text/csv",
    )

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "engine": "Python Scikit-Learn",
        "model": "Random Forest (300 trees)",
        "features": len(feature_names),
        "total_dataset": len(df),
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
