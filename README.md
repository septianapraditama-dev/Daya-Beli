# Daya-Beli
Sebuah aplikasi mengukur daya beli masyarakat DKI Jakarta terhadap komoditas pangan


## ⚙️ Bagaimana Cara Kerjanya

### 1. Parameter Affordability (Jonsson-Toole)

Sama persis dengan kode di notebook `prepro.ipynb`:

```python
JONSSON_TOOLE = 0.60   # batas rasio harga / anggaran pangan
FOOD_SHARE     = 0.55  # porsi UMP untuk kebutuhan pangan

food_budget_daily     = ump_daily * FOOD_SHARE
price_to_budget_ratio = harga_rp  / food_budget_daily
affordability_risk    = (price_to_budget_ratio > JONSSON_TOOLE)
```

### 2. Risk Score Komposit (4 Dimensi)

Dihitung di notebook preprocessing, dibaca langsung dari `affordability_flags.csv`:

| Dimensi | Kondisi | Poin |
|---------|---------|------|
| **1. Rasio JT** | > 0.60 | +2 |
| | 0.40–0.60 | +1 |
| **2. Volatilitas CV 7h** | ≥ 0.15 | +2 |
| | 0.05–0.15 | +1 |
| **3. Tren 7 hari** | > +15% | +2 |
| | +5–15% | +1 |
| **4. Posisi vs P90** | > P90 | +2 |
| | > 85% P90 | +1 |

**Total maks = 8** → Mapping:
- **0–1**: Aman 🟢
- **2–4**: Waspada 🟡
- **5–6**: Kritis 🔴

### 3. Prediksi Harga (Multi-Step Ridge)

```python
# Ambil baris terakhir per komoditas+region dari features_regression.csv
last_row = df_feat[mask].sort_values("tanggal").iloc[[-1]]
X_raw    = last_row[FEATURE_COLS].values.astype(np.float32)

for horizon in [1, 7, 14, 21, 30]:
    X_scaled = scaler_t{horizon}.transform(X_raw)
    pred_log  = ridge_t{horizon}.predict(X_scaled)[0]
    pred_rp   = np.expm1(pred_log)       # inverse log1p

    # Conformal Prediction Interval 90%
    q = conformal_q_t{horizon}[komoditas]  # q90 per komoditas
    lo_rp = np.expm1(pred_log - q)
    hi_rp = np.expm1(pred_log + q)
```

### 4. Prompt ke Gemini AI

Prompt dibangun dari:
- Harga terkini & UMP DKI harian
- Anggaran pangan (55% UMP)
- Rasio Jonsson-Toole & risk level
- Proyeksi t+1/7/14/21/30 hari + CI 90%
- MAPE model per komoditas

Gemini diminta memberikan:
1. Ringkasan situasi harga
2. Analisis keterjangkauan
3. Saran pembelian & strategi finansial
4. Peringatan & catatan penting

---


## 📦 Dependencies

| Package | Versi Minimum | Fungsi |
|---------|--------------|--------|
| streamlit | 1.32.0 | Framework UI web |
| pandas | 2.0.0 | Manipulasi data |
| numpy | 1.24.0 | Komputasi numerik |
| scikit-learn | 1.3.0 | Load model Ridge & scaler |
| joblib | 1.3.0 | Load file .pkl |
| plotly | 5.18.0 | Grafik interaktif |
---

## 🎯 Fitur Aplikasi

| Tab | Fitur |
|-----|-------|
| **Dashboard** | Harga terkini, metrik affordability, tren historis, risk score per komoditas |
| **Prediksi Harga** | Proyeksi t+1/7/14/21/30 hari, CI 90%, tabel & grafik, status affordability |
| **Saran** | Analisis situasi, saran pembelian, strategi finansial berbahasa Indonesia |
| **Evaluasi Model** | MAPE per model, per horizon, per komoditas |

