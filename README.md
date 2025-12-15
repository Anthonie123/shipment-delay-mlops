# shipment-delay-mlops
# 1. Struktur Project
<img width="2450" height="557" alt="diagram-export-12-15-2025-5_29_11-PM" src="https://github.com/user-attachments/assets/b65ab956-7f86-4eeb-9018-e4d55b8b6e79" />


# 2. Prasyarat
Sebelum mulai, pastikan sudah install:
- Python 3.11 (atau minimal 3.10)
- Git
- DVC (pip install dvc)
- MLflow (pip install mlflow)
- Docker Desktop (kalau mau menjalankan API di dalam container)
- pip sudah up to date

# 3. Cara Clone dan Setup Project
3.1 Clone repository
- git clone https://github.com/<username>/<nama-repo>.git
- cd nama-repo

3.2 Buat dan aktifkan virtual environment (opsional tapi disarankan)
- Windows (PowerShell)  
python -m venv .venv  
.\.venv\Scripts\activate

- Mac / Linux
python -m venv .venv  
source .venv/bin/activate

3.3 Install dependency  
- pip install --upgrade pip  
- pip install -r requirements.txt

# 4. Setup DVC & Data
4.1 Inisialisasi DVC (kalau belum)
- dvc init

4.2 Konfigurasi remote storage lokal (contoh)
- buat folder remote  
mkdir D:\dvc-storage
- set sebagai default remote  
dvc remote add -d local_remote D:/dvc-storage

4.3 Pull data dari remote (kalau ada) atau push pertama kali
- Kalau data sudah pernah di‑push ke remote:
python -m dvc pull
- Kalau ini pertama kali:  
dvc add data/train.csv data/train2.csv data/train3.csv data/train4.csv data/train5.csv data/train6.csv
git add data/*.csv.dvc .gitignore .dvc/config  
git commit -m "Track training data with DVC"
dvc push

# 5. Menjalankan Pipeline (Prep → Train → Eval)
5.1 Jalankan pipeline lengkap  
python -m dvc repro  
Perintah ini akan:
- Menjalankan python src/data_prep.py → menghasilkan data/processed.csv
- Menjalankan python src/train.py → training beberapa model, log ke MLflow, simpan model terbaik ke model/model_rf.pkl
- Menjalankan python src/eval.py (kalau diisi) → evaluasi model  
Setelah selesai, file model/model_rf.pkl akan berisi model terbaik yang siap untuk deployment.

# 6. Experiment Tracking dengan MLflow
6.1 Menjalankan MLflow UI  
mlflow ui --port 5000  
Lalu buka di browser:  
http://127.0.0.1:5000  
Di sini bisa dilihat:
- Daftar experiment (misalnya mlops-demo)
- Setiap run training, dengan parameter (n_estimators, max_depth, dll) dan metrik (accuracy, F1, dll)
- Perbandingan run (pilih beberapa run lalu klik “Compare”)

# 7. Menjalankan API di dalam Docker (environment terisolasi)
7.1 Build Docker image  
dari root project
docker build -t final-project-api .

7.2 Jalankan container  
docker run -p 8000:8000 final-project-api  
Buka di browser: http://127.0.0.1:8000/docs  

# 9. Contoh Request ke API
9.1 Via Swagger UI  
- Buka http://127.0.0.1:8000/docs (menggunakan postman)
- http://127.0.0.1:8000/form (menggunakan website input data)
- Pilih endpoint POST /predict
- Klik Try it out
- Isi JSON contoh sesuai schema (ShipmentFeatures)
- Klik Execute → lihat response prediksi
