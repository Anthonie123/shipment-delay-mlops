# shipment-delay-mlops
# 1. Struktur Project

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
clone repo dari GitHub
git clone https://github.com/<username>/<nama-repo>.git
cd <nama-repo>

3.2 Buat dan aktifkan virtual environment (opsional tapi disarankan)
- Windows (PowerShell)  
python -m venv .venv  
.\.venv\Scripts\activate

Mac / Linux
python -m venv .venv  
source .venv/bin/activate

3.3 Install dependency  
pip install --upgrade pip  
pip install -r requirements.txt

#4. Setup DVC & Data
4.1 Inisialisasi DVC (kalau belum)
- dvc init

4.2 Konfigurasi remote storage lokal (contoh)
- buat folder remote  
mkdir D:\dvc-storage
- set sebagai default remote  
dvc remote add -d local_remote D:/dvc-storage
