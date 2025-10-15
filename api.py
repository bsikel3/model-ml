from flask import Flask, request, jsonify, send_file
import pandas as pd
import joblib
import os
from datetime import datetime
import json
import csv
from flask_cors import CORS

app = Flask(__name__)


# Konfigurasi CORS - Pilih salah satu metode di bawah:

# METODE 1: Izinkan semua origin (untuk development)
CORS(app)

# METODE 2: Izinkan origin tertentu (lebih aman untuk production)
CORS(app, resources={r"/*": {"origins": ["http://192.168.23.50:5001", "http://192.168.23.50:4001"]}})

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

CATEGORIZED_FILE = 'categorized_reviews.csv'

try:
    model = joblib.load('model_new.pkl')
    vectorizer = joblib.load('vectorizer_new.pkl')
except Exception as e:
    model = None
    vectorizer = None
    print("⚠️ Model/vectorizer tidak ditemukan:", e)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'content' not in data:
        return jsonify({"error": "Harap sertakan field 'content'"}), 400

    text = data['content']
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    # Hitung confidence
    confidence = None
    debug_info = {}
    
    if hasattr(model, "decision_function"):
        # Untuk LinearSVC
        decision_scores = model.decision_function(X)
        debug_info['has_decision_function'] = True
        debug_info['decision_shape'] = str(decision_scores.shape)
        
        if decision_scores.ndim == 1:
            # Binary classification
            confidence = float(abs(decision_scores[0]))
            debug_info['classification_type'] = 'binary'
        else:
            # Multi-class classification
            confidence = float(max(decision_scores[0]))
            debug_info['classification_type'] = 'multi-class'
            debug_info['all_scores'] = [float(s) for s in decision_scores[0]]
    elif hasattr(model, "predict_proba"):
        # Untuk model dengan predict_proba
        proba = model.predict_proba(X)[0]
        confidence = float(proba[list(model.classes_).index(prediction)])
        debug_info['has_predict_proba'] = True
    else:
        debug_info['error'] = 'Model tidak punya decision_function atau predict_proba'

    # Simpan ke history
    new_data = pd.DataFrame({
        'content': [text],
        'primary_category': [prediction],
        'confidence': [confidence],
    })

    if os.path.exists(CATEGORIZED_FILE):
        existing = pd.read_csv(CATEGORIZED_FILE)
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        combined = new_data
    combined.to_csv(CATEGORIZED_FILE, index=False)

    return jsonify({
        "input": text,
        "predicted_category": prediction,
        "confidence": confidence,
        "debug": debug_info,
        "message": "Prediction saved to categorized_reviews.csv"
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file di request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nama file kosong"}), 400

    ext = file.filename.split('.')[-1].lower()
    if ext not in ['csv', 'xlsx']:
        return jsonify({"error": "Hanya file CSV atau XLSX yang diperbolehkan"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Baca file sesuai ekstensi
    df = pd.read_excel(filepath) if ext == 'xlsx' else pd.read_csv(filepath)

    # Validasi kolom wajib
    if 'content' not in df.columns:
        return jsonify({"error": "Kolom 'content' tidak ditemukan"}), 400

    # Tambahkan kolom 'source' jika belum ada
    if 'source' not in df.columns:
        df['source'] = 'Unknown'

    # Prediksi kategori
    X = vectorizer.transform(df['content'].astype(str))
    predictions = model.predict(X)

    # Hitung confidence untuk setiap prediksi
    confidences = []
    
    if hasattr(model, "decision_function"):
        # Untuk LinearSVC, gunakan decision_function
        decision_scores = model.decision_function(X)
        
        if decision_scores.ndim == 1:
            # Binary classification
            confidences = [float(abs(score)) for score in decision_scores]
        else:
            # Multi-class classification - ambil nilai maksimum per baris
            confidences = [float(max(scores)) for scores in decision_scores]
            
    elif hasattr(model, "predict_proba"):
        # Untuk model dengan predict_proba
        proba = model.predict_proba(X)
        confidences = [float(p[list(model.classes_).index(pred)]) for p, pred in zip(proba, predictions)]
    else:
        # Fallback jika tidak ada confidence
        confidences = [None] * len(predictions)

    # Assign hasil prediksi dan confidence
    df['primary_category'] = predictions
    df['confidence'] = confidences

    # Susun kolom hasil akhir dengan urutan yang jelas
    base_columns = ['content', 'primary_category', 'confidence']
    
    # Tambahkan kolom tambahan jika ada (score, source, dll)
    optional_columns = []
    if 'score' in df.columns:
        optional_columns.append('score')
    if 'source' in df.columns:
        optional_columns.append('source')
    
    result_columns = base_columns + optional_columns
    result_df = df[result_columns]

    # Simpan hasil klasifikasi
    result_filename = f"classified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    result_df.to_csv(result_path, index=False)

    # Update file categorized_reviews.csv
    if os.path.exists(CATEGORIZED_FILE):
        existing = pd.read_csv(CATEGORIZED_FILE)
        combined = pd.concat([existing, result_df], ignore_index=True)
    else:
        combined = result_df
    combined.to_csv(CATEGORIZED_FILE, index=False)

    # Preview data (10 baris pertama)
    preview_data = result_df.head(10).to_dict(orient='records')

    # Statistik confidence
    confidence_stats = {
        "avg_confidence": float(df['confidence'].mean()) if df['confidence'].notna().any() else None,
        "min_confidence": float(df['confidence'].min()) if df['confidence'].notna().any() else None,
        "max_confidence": float(df['confidence'].max()) if df['confidence'].notna().any() else None,
    }

    return jsonify({
        "message": "Klasifikasi berhasil",
        "result_file": result_filename,
        "total_rows": len(result_df),
        "confidence_stats": confidence_stats,
        "preview": preview_data
    })

@app.route('/weekly_trend', methods=['POST'])
def weekly_trend():
    """
    API untuk menampilkan tren jumlah keluhan konsumen selama 7 hari terakhir.
    Data masih dummy (simulasi).
    """
    # Dummy data: jumlah keluhan dari Monday - Sunday
    dummy_data = [
        {"day": "Monday", "complaints": 35},
        {"day": "Tuesday", "complaints": 50},
        {"day": "Wednesday", "complaints": 40},
        {"day": "Thursday", "complaints": 55},
        {"day": "Friday", "complaints": 60},
        {"day": "Saturday", "complaints": 30},
        {"day": "Sunday", "complaints": 25},
    ]

    return jsonify({
        "message": "Weekly complaint trend (dummy data)",
        "data": dummy_data
    }), 200


@app.route('/assign', methods=['POST'])
def assign_issue():
    data = request.get_json()
    if not all(k in data for k in ('row_index', 'new_category')):
        return jsonify({"error": "Field wajib: row_index, new_category"}), 400

    if not os.path.exists(CATEGORIZED_FILE):
        return jsonify({"error": "File categorized_reviews.csv belum ada"}), 404

    df = pd.read_csv(CATEGORIZED_FILE)
    idx = int(data['row_index'])
    if idx < 0 or idx >= len(df):
        return jsonify({"error": "Index baris di luar jangkauan"}), 400

    df.at[idx, 'primary_category'] = data['new_category']
    df.at[idx, 'manually_assigned'] = True
    df.at[idx, 'assignment_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df.to_csv(CATEGORIZED_FILE, index=False)

    return jsonify({
        "message": "Kategori berhasil diubah",
        "row_index": idx,
        "new_category": data['new_category']
    })

@app.route('/dashboard', methods=['GET'])
def dashboard():
    if not os.path.exists(CATEGORIZED_FILE):
        return jsonify({"message": "Belum ada data hasil klasifikasi"}), 200

    df = pd.read_csv(CATEGORIZED_FILE)
    total_reviews = len(df)
    avg_rating = float(df['score'].mean()) if 'score' in df.columns else None
    unique_categories = df['primary_category'].nunique() if 'primary_category' in df.columns else 0
    top_issue = df['primary_category'].mode()[0] if 'primary_category' in df.columns else None

    # Hitung rata-rata confidence
    avg_confidence = float(df['confidence'].mean()) if 'confidence' in df.columns and df['confidence'].notna().any() else None

    summary = {
        "total_reviews": total_reviews,
        "avg_rating": avg_rating,
        "avg_confidence": avg_confidence,
        "unique_categories": unique_categories,
        "top_issue": top_issue,
    }

    # Tambah distribusi kategori
    if 'primary_category' in df.columns:
        summary["category_distribution"] = df['primary_category'].value_counts().to_dict()
    else:
        summary["category_distribution"] = {}

    return jsonify(summary)

@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    file_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File tidak ditemukan"}), 404

    return send_file(file_path, as_attachment=True)


@app.route('/upload-data', methods=['POST'])
def upload_csv():
    # Ambil file CSV dari request
    file = request.files['file']
    # Baca CSV ke DataFrame
    df = pd.read_csv(file)
    # Ubah ke JSON
    data_json = df.to_dict(orient='records')
    return jsonify(data_json)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4001)
