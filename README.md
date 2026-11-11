# Prediksi Churn Pelanggan dengan Decision Tree (TA-05)

Klasifikasi churn pelanggan telekomunikasi memakai Decision Tree, lengkap dengan aplikasi web Flask yang menampilkan prediksi **dan jalur keputusan pohon** untuk setiap pelanggan.

Dataset: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Struktur proyek

```
.
├── TA05_Decision_Tree_Churn.ipynb   # EDA, pra-pemrosesan, Model A vs B, visualisasi, simpan model
├── app.py                           # Aplikasi Flask
├── templates/index.html             # Halaman form + hasil
├── static/style.css
├── data/                            # Letakkan WA_Fn-UseC_-Telco-Customer-Churn.csv di sini
├── model/churn_decision_tree.joblib # Dihasilkan oleh notebook
├── requirements.txt                 # Dihasilkan/ditimpa oleh notebook
└── Procfile                         # Untuk Render / Railway
```

## 1. Latih model (jalankan notebook)

1. Unduh CSV dari halaman Kaggle di atas, simpan ke folder `data/`.
2. Buka `TA05_Decision_Tree_Churn.ipynb` (Jupyter atau Google Colab) dan jalankan semua sel dari atas ke bawah.
3. Sel terakhir menghasilkan `model/churn_decision_tree.joblib` dan `requirements.txt` berisi versi library yang tepat.
4. Jika memakai Colab, unduh kedua file itu lalu letakkan di folder proyek ini.

## 2. Jalankan aplikasi secara lokal

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Buka http://127.0.0.1:5000

## 3. Deploy ke Render

1. Push seluruh folder ini ke repositori GitHub (pastikan `model/churn_decision_tree.joblib` ikut ter-commit).
2. Di Render: **New → Web Service**, hubungkan repositori.
3. Isi:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
4. Klik **Create Web Service**. Setelah selesai, Render memberi URL publik.

## 3b. Deploy ke Railway (alternatif)

1. **New Project → Deploy from GitHub repo**, pilih repositori ini.
2. Railway membaca `Procfile` otomatis. Setelah build selesai, buka **Settings → Networking → Generate Domain** untuk mendapat URL publik.

## Catatan

- Versi `scikit-learn` saat deployment harus sama dengan saat melatih model. Itu sebabnya notebook menulis `requirements.txt` otomatis.
- Plan gratis Render akan "tidur" saat tidak dipakai, jadi akses pertama bisa lambat beberapa detik.
- Endpoint `/health` bisa dipakai untuk memeriksa bahwa aplikasi berjalan.
# decession-tree
