# shipment-delay-mlops
1. Struktur Project
text
FINAL_PROJECT/
├─ data/
│  ├─ train.csv, train2.csv, ...  # data mentah (di-track DVC)
│  └─ processed.csv               # data hasil preprocessing (output DVC)
├─ model/
│  └─ model_rf.pkl                # model terbaik untuk serving
├─ logs/
│  └─ predictions.csv             # log request API
├─ src/
│  ├─ data_prep.py                # script preprocessing
│  ├─ train.py                    # script training + MLflow logging
│  ├─ eval.py                     # (opsional) evaluasi model
│  └─ serve.py                    # FastAPI app untuk serving
├─ .dvc/                          # config DVC
├─ dvc.yaml                       # definisi pipeline DVC (prep → train → eval)
├─ dvc.lock
├─ Dockerfile                     # definisi environment Docker
├─ requirements.txt               # dependency Python
└─ README.md
