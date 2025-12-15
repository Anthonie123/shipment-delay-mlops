import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

MODEL_PATH = Path("mlruns/models/model_rf.pkl")
PROC_DATA_PATH = Path("data/processed.csv")
LOG_PATH = Path("logs/predictions.csv")
LOG_PATH.parent.mkdir(exist_ok=True)

with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)

df_proc = pd.read_csv(PROC_DATA_PATH)
TRAIN_FEATURE_COLUMNS = df_proc.drop("Reached.on.Time_Y.N", axis=1).columns.tolist()


class ShipmentFeatures(BaseModel):
    Warehouse_block: str
    Mode_of_Shipment: str
    Customer_care_calls: int
    Customer_rating: int
    Cost_of_the_Product: int
    Prior_purchases: int
    Product_importance: str
    Gender: str
    Discount_offered: int
    Weight_in_gms: int


def preprocess(data: ShipmentFeatures) -> pd.DataFrame:
    df = pd.DataFrame([data.dict()])
    df_enc = pd.get_dummies(df)
    for col in TRAIN_FEATURE_COLUMNS:
        if col not in df_enc.columns:
            df_enc[col] = 0
    df_enc = df_enc[TRAIN_FEATURE_COLUMNS]
    return df_enc


def log_prediction(features: ShipmentFeatures, prediction: int):
    row = features.dict()
    row["prediction"] = prediction
    row["timestamp"] = datetime.now().isoformat()

    if LOG_PATH.exists():
        df_log = pd.read_csv(LOG_PATH)
        df_log = pd.concat([df_log, pd.DataFrame([row])], ignore_index=True)
    else:
        df_log = pd.DataFrame([row])

    df_log.to_csv(LOG_PATH, index=False)


@app.post("/predict")
def predict(features: ShipmentFeatures):
    X = preprocess(features)
    y_pred = int(model.predict(X)[0])
    log_prediction(features, y_pred)
    return {"prediction": y_pred}


# ------------- UI: Form -------------
BASE_STYLE = """
    <style>
      body {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: linear-gradient(135deg, #0f172a, #1e293b);
        margin: 0;
        padding: 0;
        color: #e5e7eb;
      }
      .nav {
        padding: 14px 32px;
        background: rgba(15,23,42,0.9);
        border-bottom: 1px solid #1f2937;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }
      .nav-title {
        font-weight: 600;
        letter-spacing: .03em;
      }
      .nav-links a {
        color: #9ca3af;
        margin-left: 16px;
        text-decoration: none;
        font-size: 14px;
      }
      .nav-links a:hover {
        color: #e5e7eb;
      }
      .container {
        max-width: 900px;
        margin: 32px auto;
        padding: 0 16px 40px 16px;
      }
      .card {
        background: rgba(15,23,42,0.95);
        border-radius: 16px;
        padding: 24px 24px 28px 24px;
        box-shadow: 0 18px 45px rgba(15,23,42,0.9);
        border: 1px solid #1f2937;
      }
      h2 {
        margin-top: 0;
        margin-bottom: 4px;
        font-size: 22px;
      }
      .subtitle {
        font-size: 13px;
        color: #9ca3af;
        margin-bottom: 20px;
      }
      form {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 16px 24px;
      }
      .field {
        display: flex;
        flex-direction: column;
        font-size: 13px;
      }
      .field label {
        margin-bottom: 4px;
        color: #d1d5db;
      }
      .field input {
        padding: 7px 10px;
        border-radius: 9px;
        border: 1px solid #374151;
        background: #020617;
        color: #e5e7eb;
        font-size: 13px;
        outline: none;
      }
      .field input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 1px rgba(99,102,241,0.6);
      }
      .hint {
        font-size: 11px;
        color: #9ca3af;
        margin-top: 2px;
      }
      .actions {
        grid-column: 1 / -1;
        margin-top: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      .btn {
        padding: 8px 18px;
        border-radius: 999px;
        border: none;
        cursor: pointer;
        font-size: 13px;
        font-weight: 500;
      }
      .btn-primary {
        background: linear-gradient(135deg, #4f46e5, #6366f1);
        color: white;
      }
      .btn-primary:hover {
        filter: brightness(1.05);
      }
      .btn-link {
        background: transparent;
        color: #9ca3af;
        text-decoration: none;
      }
      .btn-link:hover {
        color: #e5e7eb;
      }
      .pill {
        display: inline-flex;
        align-items: center;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 11px;
        background: rgba(22,163,74,0.15);
        color: #bbf7d0;
        border: 1px solid rgba(22,163,74,0.5);
      }
      table {
        border-collapse: collapse;
        width: 100%;
        margin-top: 10px;
        background: #020617;
      }
      th, td {
        border: 1px solid #1f2937;
        padding: 6px 8px;
        font-size: 11px;
      }
      th {
        background: #030712;
      }
      tr:nth-child(even) td {
        background: #020617;
      }
      tr:nth-child(odd) td {
        background: #020617;
      }
      .result {
        margin-top: 16px;
        padding: 10px 14px;
        border-radius: 10px;
        background: rgba(37,99,235,0.15);
        border: 1px solid rgba(59,130,246,0.6);
        font-size: 13px;
      }
    </style>
"""


@app.get("/form", response_class=HTMLResponse)
def form_page():
    html = f"""
    <html>
      <head>
        <title>Form Prediksi Pengiriman</title>
        {BASE_STYLE}
      </head>
      <body>
        <div class="nav">
          <div class="nav-title">Delivery ML Service</div>
          <div class="nav-links">
            <a href="/form">Form Prediksi</a>
            <a href="/monitor">Monitoring</a>
          </div>
        </div>

        <div class="container">
          <div class="card">
            <h2>Input Data Pengiriman</h2>
            <div class="subtitle">
              Masukkan detail paket dan pengiriman, lalu sistem akan memprediksi apakah paket akan sampai tepat waktu.
            </div>

            <form method="post">
              <div class="field">
                <label>Warehouse_block</label>
                <input name="Warehouse_block" value="D" />
              </div>

              <div class="field">
                <label>Mode_of_Shipment</label>
                <input name="Mode_of_Shipment" value="Flight" />
              </div>

              <div class="field">
                <label>Customer_care_calls (1–5)</label>
                <input type="number" name="Customer_care_calls" value="3" min="1" max="5" />
                <div class="hint">Berapa kali pelanggan menghubungi customer service.</div>
              </div>

              <div class="field">
                <label>Customer_rating (1–5)</label>
                <input type="number" name="Customer_rating" value="4" min="1" max="5" />
                <div class="hint">Rating kepuasan pelanggan.</div>
              </div>

              <div class="field">
                <label>Cost_of_the_Product</label>
                <input type="number" name="Cost_of_the_Product" value="200" />
              </div>

              <div class="field">
                <label>Prior_purchases</label>
                <input type="number" name="Prior_purchases" value="2" />
              </div>

              <div class="field">
                <label>Product_importance</label>
                <input name="Product_importance" value="medium" />
              </div>

              <div class="field">
                <label>Gender</label>
                <input name="Gender" value="M" />
              </div>

              <div class="field">
                <label>Discount_offered</label>
                <input type="number" name="Discount_offered" value="10" />
              </div>

              <div class="field">
                <label>Weight_in_gms</label>
                <input type="number" name="Weight_in_gms" value="3000" />
              </div>

              <div class="actions">
                <button type="submit" class="btn btn-primary">Prediksi</button>
                <a href="/monitor" class="btn btn-link">Lihat Monitoring &raquo;</a>
              </div>
            </form>
          </div>
        </div>
      </body>
    </html>
    """
    return html


@app.post("/form", response_class=HTMLResponse)
def form_submit(
    Warehouse_block: str = Form(...),
    Mode_of_Shipment: str = Form(...),
    Customer_care_calls: int = Form(...),
    Customer_rating: int = Form(...),
    Cost_of_the_Product: int = Form(...),
    Prior_purchases: int = Form(...),
    Product_importance: str = Form(...),
    Gender: str = Form(...),
    Discount_offered: int = Form(...),
    Weight_in_gms: int = Form(...),
):
    Customer_care_calls_clipped = max(1, min(5, Customer_care_calls))
    Customer_rating_clipped = max(1, min(5, Customer_rating))

    features = ShipmentFeatures(
        Warehouse_block=Warehouse_block,
        Mode_of_Shipment=Mode_of_Shipment,
        Customer_care_calls=Customer_care_calls_clipped,
        Customer_rating=Customer_rating_clipped,
        Cost_of_the_Product=Cost_of_the_Product,
        Prior_purchases=Prior_purchases,
        Product_importance=Product_importance,
        Gender=Gender,
        Discount_offered=Discount_offered,
        Weight_in_gms=Weight_in_gms,
    )

    X = preprocess(features)
    y_pred = int(model.predict(X)[0])
    log_prediction(features, y_pred)

    html = f"""
    <html>
      <head>
        <title>Hasil Prediksi</title>
        {BASE_STYLE}
      </head>
      <body>
        <div class="nav">
          <div class="nav-title">Delivery ML Service</div>
          <div class="nav-links">
            <a href="/form">Form Prediksi</a>
            <a href="/monitor">Monitoring</a>
          </div>
        </div>

        <div class="container">
          <div class="card">
            <h2>Hasil Prediksi</h2>
            <div class="subtitle">
              Berdasarkan input yang kamu berikan, model memberikan hasil berikut.
            </div>
            <div class="result">
              Prediksi keterlambatan (Reached.on.Time_Y.N) = <b>{y_pred}</b>
            </div>
            <div style="margin-top:14px; font-size:13px;">
              <a href="/form" class="btn btn-link">&laquo; Kembali ke Form</a>
              <a href="/monitor" class="btn btn-link">Lihat Monitoring</a>
            </div>
          </div>
        </div>
      </body>
    </html>
    """
    return html


# ------------- UI: Monitoring -------------
@app.get("/monitor", response_class=HTMLResponse)
def monitor():
    if not LOG_PATH.exists():
        html = f"""
        <html>
          <head>
            <title>Monitoring Prediksi</title>
            {BASE_STYLE}
          </head>
          <body>
            <div class="nav">
              <div class="nav-title">Delivery ML Service</div>
              <div class="nav-links">
                <a href="/form">Form Prediksi</a>
                <a href="/monitor">Monitoring</a>
              </div>
            </div>
            <div class="container">
              <div class="card">
                <h2>Monitoring Prediksi</h2>
                <div class="subtitle">Belum ada data prediksi yang masuk.</div>
                <a href="/form" class="btn btn-primary">Buat Prediksi Pertama</a>
              </div>
            </div>
          </body>
        </html>
        """
        return html

    df = pd.read_csv(LOG_PATH)
    total = len(df)
    latest = df.tail(50).copy()

    # hitung ringkasan sederhana
    late_ratio = latest["prediction"].mean() if "prediction" in latest.columns else 0
    ontime_pct = (1 - late_ratio) * 100
    late_pct = late_ratio * 100

    # ubah kolom prediction jadi badge teks
    def badge(val):
        if int(val) == 1:
            return '<span style="color:#fecaca; background:rgba(220,38,38,0.18); padding:2px 6px; border-radius:999px; font-size:11px;">Terlambat (1)</span>'
        else:
            return '<span style="color:#bbf7d0; background:rgba(22,163,74,0.18); padding:2px 6px; border-radius:999px; font-size:11px;">Tepat Waktu (0)</span>'

    latest_display = latest.copy()
    if "prediction" in latest_display.columns:
        latest_display["prediction"] = latest_display["prediction"].apply(badge)

        html_table = latest_display.to_html(
        classes="table",
        index=False,
        escape=False  # supaya HTML badge tidak di-escape
    )

    html = f"""
    <html>
      <head>
        <title>Monitoring Prediksi</title>
        {BASE_STYLE}
      </head>
      <body>
        <div class="nav">
          <div class="nav-title">Delivery ML Service</div>
          <div class="nav-links">
            <a href="/form">Form Prediksi</a>
            <a href="/monitor">Monitoring</a>
          </div>
        </div>

        <div class="container">
          <div class="card" style="max-width: 1100px; margin: 0 auto;">
            <h2>Monitoring Prediksi</h2>
            <div class="subtitle">
              Ringkasan aktivitas API model berdasarkan log prediksi yang tersimpan.
            </div>

            <div style="display:flex; gap:16px; margin-bottom:18px; flex-wrap:wrap;">
              <div style="flex:1; min-width:180px; padding:10px 12px; border-radius:12px; background:rgba(15,23,42,0.9); border:1px solid #1f2937;">
                <div style="font-size:11px; color:#9ca3af;">Total request tersimpan</div>
                <div style="font-size:18px; font-weight:600;">{total}</div>
              </div>
              <div style="flex:1; min-width:180px; padding:10px 12px; border-radius:12px; background:rgba(15,23,42,0.9); border:1px solid #1f2937;">
                <div style="font-size:11px; color:#9ca3af;">Tepat waktu (prediksi 0)</div>
                <div style="font-size:18px; font-weight:600; color:#bbf7d0;">{ontime_pct:.1f}%</div>
              </div>
              <div style="flex:1; min-width:180px; padding:10px 12px; border-radius:12px; background:rgba(15,23,42,0.9); border:1px solid #1f2937;">
                <div style="font-size:11px; color:#9ca3af;">Terlambat (prediksi 1)</div>
                <div style="font-size:18px; font-weight:600; color:#fecaca;">{late_pct:.1f}%</div>
              </div>
            </div>

            <div class="subtitle" style="margin-top:8px;">50 request terbaru</div>
            <div style="overflow-x:auto; border-radius:12px; border:1px solid #1f2937;">
              {html_table}
            </div>
          </div>
        </div>
      </body>
    </html>
    """
    return html
