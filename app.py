import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import date
import plotly.graph_objects as go
import plotly.express as px



st.set_page_config(
    page_title="Prediksi Harga Pangan DKI Jakarta",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

JONSSON_TOOLE = 0.60
FOOD_SHARE    = 0.55

RISK_COLOR = {
    "aman":    "#16A34A",
    "waspada": "#D97706",
    "kritis":  "#DC2626",
}
RISK_LABEL = {
    "aman":    "Aman 🟢",
    "waspada": "Waspada 🟡",
    "kritis":  "Kritis 🔴",
}
KOMODITAS_LABEL = {
    "ayam_ras":             "Ayam Ras",
    "bawang_merah":         "Bawang Merah",
    "bawang_putih":         "Bawang Putih",
    "beras_premium":        "Beras Premium",
    "cabai_merah_keriting": "Cabai Merah Keriting",
    "cabai_rawit_merah":    "Cabai Rawit Merah",
    "daging_sapi":          "Daging Sapi",
    "gula_pasir":           "Gula Pasir",
    "telur_ayam":           "Telur Ayam",
}
HORIZONS = [1, 7, 14, 21, 30]
BASE_DIR  = os.path.dirname(__file__)


# ── RISK SCORE ──
def hitung_risk_score(row):
    skor = 0
    jt = row.get("price_to_budget_ratio", 0)
    if jt > 0.60: skor += 2
    elif jt > 0.40: skor += 1
    std = row.get("std_7_log", 0); ma = row.get("ma_7_log", 1)
    cv = std / (abs(ma) + 1e-8)
    if cv >= 0.15: skor += 2
    elif cv >= 0.05: skor += 1
    delta = row.get("pct_change_7d", 0)
    if delta > 0.15: skor += 2
    elif delta > 0.05: skor += 1
    p90 = row.get("harga_p90", None); hrp = row.get("harga_rp", 0)
    if p90 is not None and hrp > p90: skor += 2
    elif p90 is not None and hrp > p90 * 0.85: skor += 1
    return int(skor)

def risk_score_to_level(skor):
    if skor >= 6: return "kritis"
    if skor >= 2: return "waspada"
    return "aman"


# ── LOAD DATA ──
@st.cache_resource
def load_artifacts():
    artifacts_dir = os.path.join(BASE_DIR, "artifacts")
    models_dir    = os.path.join(BASE_DIR, "models")
    with open(os.path.join(artifacts_dir, "feature_cols.json")) as f:
        meta = json.load(f)
    le_kom = joblib.load(os.path.join(artifacts_dir, "le_komoditas.pkl"))
    le_reg = joblib.load(os.path.join(artifacts_dir, "le_region.pkl"))
    multi_models = {}; multi_scalers = {}; conformal_q = {}
    for h in HORIZONS:
        multi_models[h]  = joblib.load(os.path.join(models_dir, f"ridge_t{h}.pkl"))
        multi_scalers[h] = joblib.load(os.path.join(models_dir, f"scaler_t{h}.pkl"))
        conformal_q[h]   = joblib.load(os.path.join(models_dir, f"conformal_q_t{h}.pkl"))
    with open(os.path.join(models_dir, "evaluation_report.json")) as f:
        eval_report = json.load(f)
    return meta, le_kom, le_reg, multi_models, multi_scalers, conformal_q, eval_report

@st.cache_data
def load_data():
    processed_dir = os.path.join(BASE_DIR, "processed")
    df_feat = pd.read_csv(os.path.join(processed_dir, "features_regression.csv"), parse_dates=["tanggal"])
    df_feat.columns = df_feat.columns.str.strip()
    df_feat = df_feat.sort_values("tanggal").reset_index(drop=True)
    p90 = df_feat.groupby(["komoditas","region"])["harga_rp"].quantile(0.90).rename("harga_p90").reset_index()
    df_feat = df_feat.merge(p90, on=["komoditas","region"], how="left")
    df_feat["risk_score"] = df_feat.apply(hitung_risk_score, axis=1)
    df_feat["risk_level"] = df_feat["risk_score"].apply(risk_score_to_level)
    aff_cols = ["tanggal","komoditas","region","harga_rp","ump_daily","food_budget_daily",
                "price_to_budget_ratio","affordability_risk","harga_p90","risk_score","risk_level",
                "std_7_log","ma_7_log","pct_change_7d"]
    df_aff = df_feat[[c for c in aff_cols if c in df_feat.columns]].copy()
    return df_aff, df_feat

def get_latest_per_komoditas(df_aff, region):
    sub = df_aff[df_aff["region"] == region].copy()
    if sub.empty: return pd.DataFrame()
    idx = sub.groupby("komoditas")["tanggal"].idxmax()
    return sub.loc[idx].reset_index(drop=True)


# ── PREDIKSI ──
def predict_mingguan(komoditas, region, df_feat, meta, multi_models, multi_scalers, conformal_q):
    FEATURE_COLS = meta["feature_cols"]
    mask = (df_feat["komoditas"] == komoditas) & (df_feat["region"] == region)
    sub  = df_feat[mask].dropna(subset=FEATURE_COLS).sort_values("tanggal")
    if sub.empty: return None
    X_raw = sub.iloc[[-1]][FEATURE_COLS].values.astype(np.float32)
    today = pd.Timestamp(date.today())
    X_s1 = multi_scalers[1].transform(X_raw); X_s7 = multi_scalers[7].transform(X_raw)
    pl_t1 = float(multi_models[1].predict(X_s1)[0]); pl_t7 = float(multi_models[7].predict(X_s7)[0])
    q_t1 = conformal_q[1].get(komoditas, conformal_q[1].get("__global__", 0.1))
    q_t7 = conformal_q[7].get(komoditas, conformal_q[7].get("__global__", 0.1))
    rows = []
    for day in range(1, 8):
        alpha = (day - 1) / 6.0
        pl = pl_t1 * (1 - alpha) + pl_t7 * alpha
        q  = q_t1  * (1 - alpha) + q_t7  * alpha
        rows.append({"horizon": day, "tanggal": today + pd.Timedelta(days=day),
                     "pred_rp": max(float(np.expm1(pl)), 0),
                     "lower_rp": max(float(np.expm1(pl - q)), 0),
                     "upper_rp": max(float(np.expm1(pl + q)), 0)})
    return pd.DataFrame(rows)

def predict_bulanan(komoditas, region, df_feat, meta, multi_models, multi_scalers, conformal_q):
    FEATURE_COLS = meta["feature_cols"]
    mask = (df_feat["komoditas"] == komoditas) & (df_feat["region"] == region)
    sub  = df_feat[mask].dropna(subset=FEATURE_COLS).sort_values("tanggal")
    if sub.empty: return None
    X_raw = sub.iloc[[-1]][FEATURE_COLS].values.astype(np.float32)
    today = pd.Timestamp(date.today())
    label_map = {7:"Minggu 1", 14:"Minggu 2", 21:"Minggu 3", 30:"Minggu 4"}
    rows = []
    for h in [7, 14, 21, 30]:
        X_s = multi_scalers[h].transform(X_raw)
        pl  = float(multi_models[h].predict(X_s)[0])
        q   = conformal_q[h].get(komoditas, conformal_q[h].get("__global__", 0.1))
        rows.append({"horizon": h, "label": label_map[h],
                     "tanggal": today + pd.Timedelta(days=h),
                     "pred_rp": max(float(np.expm1(pl)), 0),
                     "lower_rp": max(float(np.expm1(pl - q)), 0),
                     "upper_rp": max(float(np.expm1(pl + q)), 0)})
    return pd.DataFrame(rows)

def hitung_affordability(pred_rp, gaji_harian):
    food_budget = gaji_harian * FOOD_SHARE
    ratio = pred_rp / food_budget if food_budget > 0 else 0
    return {"food_budget": food_budget, "ratio": ratio, "affordability_ok": ratio <= JONSSON_TOOLE}


# ── ANALISIS CERDAS (LOCAL AI) ──
def generate_saran_pintar(komoditas_label, region, harga_kini, gaji_harian,
                           df_pred, risk_level, risk_score, mode, eval_report):
    """
    Menghasilkan analisis harga yang personal, adaptif, dan terasa natural —
    tanpa API eksternal. Sepenuhnya berbasis data prediksi dan konteks pengguna.
    """
    food_budget  = gaji_harian * FOOD_SHARE
    gaji_bulanan = gaji_harian * 30
    ratio_kini   = harga_kini / food_budget if food_budget > 0 else 0
    today_str    = date.today().strftime("%d %B %Y")

    # ── Akurasi model per komoditas ──
    per_kom  = {k["komoditas"]: k for k in eval_report.get("per_komoditas", [])}
    kom_key  = komoditas_label.lower().replace(" ", "_")
    mape_kom = per_kom.get(kom_key, {}).get("MAPE", eval_report["best_model"]["MAPE"])

    # ── Analisis tren harga ──
    pred_vals  = df_pred["pred_rp"].tolist()
    last_pred  = pred_vals[-1]
    max_pred   = max(pred_vals)
    min_pred   = min(pred_vals)
    delta_pct  = (last_pred - harga_kini) / harga_kini * 100 if harga_kini > 0 else 0

    # Arah tren
    if delta_pct > 5:
        tren = "naik"
    elif delta_pct < -5:
        tren = "turun"
    else:
        tren = "stabil"

    # Volatilitas intra-periode
    spread_pct = (max_pred - min_pred) / harga_kini * 100 if harga_kini > 0 else 0
    is_volatile_period = spread_pct > 10

    # ── Analisis keterjangkauan ──
    affs       = [hitung_affordability(r["pred_rp"], gaji_harian) for _, r in df_pred.iterrows()]
    n_over     = sum(1 for a in affs if not a["affordability_ok"])
    n_total    = len(affs)
    max_ratio  = max(a["ratio"] for a in affs)
    biaya_total = sum(r["pred_rp"] for _, r in df_pred.iterrows())
    period_label = "7 hari" if mode == "minggu" else "4 minggu"

    # ── Konteks komoditas ──
    alternatif_map = {
        "Ayam Ras":            "ikan segar (bandeng, lele, kembung), tahu, atau tempe",
        "Bawang Merah":        "bawang Brebes yang umumnya lebih murah, atau bawang bombay untuk masakan berkuah",
        "Bawang Putih":        "bawang putih kering/bubuk, atau bawang lokal yang lebih terjangkau",
        "Beras Premium":       "beras medium kualitas baik, atau beras merah yang lebih bergizi",
        "Cabai Merah Keriting":"cabai kering, cabai rawit hijau, atau sambal kemasan bila harga terlalu tinggi",
        "Cabai Rawit Merah":   "cabai rawit hijau, cabai merah besar, atau sambal kemasan yang stabil harganya",
        "Daging Sapi":         "daging ayam, ikan, tempe, atau tahu sebagai sumber protein yang lebih terjangkau",
        "Gula Pasir":          "gula aren, gula kelapa, atau kurangi penggunaan gula bertahap",
        "Telur Ayam":          "tahu, tempe, atau ikan teri sebagai alternatif protein padat gizi",
    }
    alternatif   = alternatif_map.get(komoditas_label, "bahan pokok lain yang tersedia di pasar")
    volatile_kom = {"Cabai Rawit Merah", "Cabai Merah Keriting", "Bawang Merah", "Bawang Putih"}
    is_volatile  = komoditas_label in volatile_kom

    # ── Konteks gaji ──
    if gaji_bulanan < 3_500_000:
        bracket      = "rendah"
        budget_note  = "Dengan kondisi anggaran yang cukup ketat"
        tip_belanja  = (
            "**Tips hemat:** Datang ke pasar tradisional pagi hari (07:00–09:00) — "
            "harga biasanya 10–20% lebih murah. Bawa tas sendiri dan beli langsung dari pedagang, "
            "hindari supermarket untuk bahan basah."
        )
    elif gaji_bulanan < 8_000_000:
        bracket      = "menengah"
        budget_note  = "Dengan penghasilan Anda saat ini"
        tip_belanja  = (
            "**Strategi belanja mingguan:** Rencanakan menu 7 hari sekaligus, "
            "lalu belanja sekali di pasar induk untuk dapat harga grosir yang lebih hemat "
            "dibanding belanja harian."
        )
    else:
        bracket      = "tinggi"
        budget_note  = "Dengan penghasilan Anda yang relatif memadai"
        tip_belanja  = (
            "**Diversifikasi sumber pangan:** Variasikan protein dan bahan pokok agar tidak "
            "bergantung pada satu komoditas — ini juga lebih baik untuk keseimbangan gizi keluarga."
        )

    # ── Narasi situasi berdasarkan tren ──
    if tren == "naik":
        narasi_tren = (
            f"Proyeksi model menunjukkan kecenderungan **naik sekitar {abs(delta_pct):.1f}%** "
            f"dalam {period_label} ke depan, dengan harga diperkirakan mencapai kisaran "
            f"**Rp{last_pred:,.0f}** di akhir periode."
        )
        narasi_tambahan = (
            f"Kenaikan ini {'tergolong signifikan dan perlu diwaspadai' if abs(delta_pct) > 15 else 'masih dalam batas wajar'}. "
            + (f"Sebagai komoditas yang harganya mudah bergejolak, {komoditas_label} sangat sensitif "
               f"terhadap perubahan cuaca dan musim panen." if is_volatile else
               f"Pergerakan harga {komoditas_label} cenderung mengikuti pola permintaan musiman dan biaya distribusi.")
        )
    elif tren == "turun":
        narasi_tren = (
            f"Proyeksi model menunjukkan kecenderungan **turun sekitar {abs(delta_pct):.1f}%** "
            f"dalam {period_label} ke depan — harga diperkirakan bergerak menuju "
            f"**Rp{last_pred:,.0f}**."
        )
        narasi_tambahan = (
            f"Ini adalah sinyal positif. "
            + (f"Namun perlu diingat bahwa {komoditas_label} tergolong volatile — "
               f"penurunan ini bisa berbalik arah dengan cepat jika ada gangguan pasokan." if is_volatile else
               f"Penurunan harga ini kemungkinan didorong oleh peningkatan pasokan atau penurunan permintaan musiman.")
        )
    else:
        narasi_tren = (
            f"Harga diperkirakan **relatif stabil** dalam {period_label} ke depan, "
            f"bergerak di rentang **Rp{min_pred:,.0f} – Rp{max_pred:,.0f}**."
        )
        narasi_tambahan = (
            f"Kondisi ini memberikan kepastian yang cukup baik untuk perencanaan belanja. "
            + (f"Meski demikian, harga {komoditas_label} bisa bergerak tiba-tiba jika ada faktor eksternal "
               f"seperti perubahan cuaca atau kebijakan impor." if is_volatile else
               f"Stabilitas ini umumnya mencerminkan keseimbangan antara pasokan dan permintaan di pasar.")
        )

    # ── Narasi keterjangkauan ──
    if ratio_kini <= JONSSON_TOOLE:
        narasi_aff_kini = (
            f"Saat ini harga menyerap **{ratio_kini*100:.0f}% dari anggaran makan** Anda "
            f"(batas aman: 60%) — masih dalam zona aman."
        )
    else:
        narasi_aff_kini = (
            f"Harga sudah menyerap **{ratio_kini*100:.0f}% dari anggaran makan** Anda "
            f"— sudah melampaui batas aman 60%."
        )

    if n_over == 0:
        narasi_aff_pred = (
            f"✅ Seluruh proyeksi {period_label} masih dalam batas anggaran. "
            f"Harga tertinggi yang diprediksi (Rp{max_pred:,.0f}) setara "
            f"**{max_ratio*100:.0f}%** dari anggaran makan harian Anda."
        )
    elif n_over < n_total:
        worst_row = df_pred.iloc[
            max(range(n_total), key=lambda i: affs[i]["ratio"])
        ]
        label_worst = (
            f"Hari ke-{int(worst_row['horizon'])}" if mode == "minggu"
            else worst_row.get("label", f"Horizon {worst_row['horizon']}")
        )
        narasi_aff_pred = (
            f"⚠️ **{n_over} dari {n_total} periode** diperkirakan melampaui batas anggaran. "
            f"Periode paling berat adalah **{label_worst}** dengan prediksi harga "
            f"Rp{worst_row['pred_rp']:,.0f} ({max_ratio*100:.0f}% dari anggaran makan). "
            f"{budget_note}, perlu sedikit penyesuaian pola belanja pada periode tersebut."
        )
    else:
        narasi_aff_pred = (
            f"🔴 **Seluruh {n_total} periode** melebihi batas anggaran makan Anda, "
            f"dengan rasio tertinggi mencapai **{max_ratio*100:.0f}%**. "
            f"{budget_note}, komoditas ini saat ini benar-benar memberatkan — "
            f"sangat direkomendasikan untuk mempertimbangkan alternatif."
        )

    # ── Saran pembelian adaptif ──
    saran_beli = []

    if tren == "naik":
        saran_beli.append(
            f"**Beli dalam 1–2 hari ke depan.** "
            f"Harga diperkirakan naik {abs(delta_pct):.1f}% — menunda berarti membayar lebih mahal. "
            f"{'Beli stok secukupnya (3–5 hari) untuk mengunci harga sekarang.' if n_over == 0 else 'Cukup beli kebutuhan 2–3 hari agar tidak terlalu membebani anggaran.'}"
        )
    elif tren == "turun":
        saran_beli.append(
            f"**Tunda pembelian stok besar.** "
            f"Harga sedang dalam tren turun {abs(delta_pct):.1f}%. "
            f"Beli secukupnya untuk hari ini saja, lalu stok besar di akhir periode saat harga sudah lebih rendah."
        )
    else:
        saran_beli.append(
            f"**Beli sesuai kebutuhan rutin.** "
            f"Harga stabil — tidak ada urgensi untuk menimbun maupun menunda. "
            f"Pola belanja mingguan yang teratur sudah optimal."
        )

    if is_volatile_period:
        saran_beli.append(
            f"**Waspadai fluktuasi harian.** "
            f"Model memperkirakan rentang harga yang cukup lebar (Rp{min_pred:,.0f}–Rp{max_pred:,.0f}) "
            f"— harga bisa bergerak cukup signifikan antarhari. Pantau harga pasar sebelum memutuskan beli stok."
        )

    if n_over > 0 or risk_level in ("waspada", "kritis"):
        saran_beli.append(
            f"**Pertimbangkan substitusi bahan:** {alternatif}. "
            f"Alternatif ini umumnya lebih terjangkau dan tetap memenuhi kebutuhan gizi yang setara."
        )

    saran_beli.append(tip_belanja)

    # ── Catatan penting ──
    catatan = []
    catatan.append(
        f"**Akurasi model:** Rata-rata kesalahan prediksi untuk {komoditas_label} adalah "
        f"**{mape_kom:.1f}%** — harga nyata bisa berbeda ±{mape_kom:.1f}% dari angka di atas. "
        f"{'Karena komoditas ini volatile, ketidakpastian lebih tinggi untuk horizon > 14 hari.' if is_volatile else 'Prediksi jangka pendek (1–7 hari) umumnya lebih akurat.'}"
    )
    catatan.append(
        f"**Faktor eksternal** seperti hari raya besar, cuaca ekstrem, bencana alam, "
        f"atau kebijakan pemerintah (HET, subsidi, larangan impor) dapat mengubah harga "
        f"secara tiba-tiba di luar jangkauan model statistik."
    )
    catatan.append(
        f"**Kisaran harga** (area biru pada grafik) dihitung dengan metode *Conformal Prediction* — "
        f"90% kemungkinan harga nyata jatuh di rentang tersebut berdasarkan pola residual historis."
    )

    # ── Susun output markdown ──
    risk_icon = {"aman": "🟢", "waspada": "🟡", "kritis": "🔴"}.get(risk_level, "⚪")
    risk_desc = {
        "aman":    "harga dalam kisaran wajar dan stabil",
        "waspada": "harga mulai memberatkan atau menunjukkan tren naik",
        "kritis":  "harga sudah melampaui batas aman — perlu perhatian segera",
    }.get(risk_level, "")

    out = []
    out.append(f"### 🤖 Analisis Harga: {komoditas_label} — {region}")
    out.append(f"*Dibuat otomatis · {today_str} · Model: Ridge Regression ({mape_kom:.1f}% MAPE)*")
    out.append("")
    out.append("---")
    out.append("")

    # Bagian 1
    out.append("#### 📊 Situasi Harga Terkini")
    out.append("")
    out.append(
        f"Harga **{komoditas_label}** di **{region}** saat ini **Rp{harga_kini:,.0f}**. "
        + narasi_tren
    )
    out.append("")
    out.append(narasi_tambahan)
    out.append("")
    out.append(
        f"> {risk_icon} **Tingkat Risiko: {risk_level.upper()} (Skor {risk_score}/8)** "
        f"— {risk_desc}."
    )
    out.append("")

    # Bagian 2
    out.append("---")
    out.append("#### 💰 Keterjangkauan untuk Penghasilan Anda")
    out.append("")
    out.append(
        f"Gaji bulanan Anda **Rp{gaji_bulanan:,.0f}** → anggaran makan harian "
        f"**Rp{food_budget:,.0f}/hari** (55% dari gaji harian). {narasi_aff_kini}"
    )
    out.append("")
    out.append(narasi_aff_pred)
    if mode == "minggu":
        out.append("")
        out.append(
            f"📌 Estimasi total pengeluaran {komoditas_label} selama 7 hari ke depan: "
            f"**±Rp{biaya_total:,.0f}** (asumsi 1 unit/hari)."
        )
    out.append("")

    # Bagian 3
    out.append("---")
    out.append("#### 🛒 Saran Pembelian")
    out.append("")
    for i, s in enumerate(saran_beli, 1):
        out.append(f"{i}. {s}")
        out.append("")

    # Bagian 4
    out.append("---")
    out.append("#### 📌 Hal yang Perlu Diperhatikan")
    out.append("")
    for c in catatan:
        out.append(f"- {c}")
    out.append("")
    out.append("---")
    out.append(
        "*⚠️ Analisis ini dihasilkan secara otomatis berdasarkan proyeksi model statistik. "
        "Gunakan sebagai referensi tambahan — bukan pengganti penilaian pribadi Anda.*"
    )

    return "\n".join(out)


# ── SIDEBAR ──
def render_sidebar(tanggal_data_terakhir=None):
    with st.sidebar:
        st.title("🛒 Pangan DKI")
        st.markdown("---")
        st.subheader("🥦 Komoditas")
        komoditas_sel = st.selectbox("Komoditas", list(KOMODITAS_LABEL.keys()),
                                     format_func=lambda x: KOMODITAS_LABEL[x],
                                     label_visibility="collapsed")
        st.subheader("📍 Wilayah")
        region_sel = st.selectbox("Wilayah",
                                  ["Jakarta Barat","Jakarta Pusat","Jakarta Selatan",
                                   "Jakarta Timur","Jakarta Utara"],
                                  label_visibility="collapsed")
        st.markdown("---")
        st.subheader("💼 Gaji Bulanan Anda")
        st.caption("Masukkan penghasilan bulanan untuk menghitung keterjangkauan harga.")
        gaji_bulanan = st.number_input("Gaji", min_value=100_000, max_value=100_000_000,
                                       value=5_067_381, step=100_000, format="%d",
                                       label_visibility="collapsed")
        st.markdown(
            f"<div class='gaji-card'>"
            f"<div class='gaji-label'>GAJI BULANAN</div>"
            f"<div class='gaji-value'>Rp {gaji_bulanan:,.0f}</div>"
            f"<div class='gaji-sub'>per bulan</div></div>",
            unsafe_allow_html=True)
        gaji_harian = gaji_bulanan / 30
        st.caption(f"📅 Setara **Rp{gaji_harian:,.0f} / hari**")
        st.caption(f"🍱 Anggaran makan: **Rp{gaji_harian * FOOD_SHARE:,.0f} / hari**")
        st.markdown("---")
        st.caption(f"📅 Hari ini: **{date.today().strftime('%d %b %Y')}**")
        st.caption("🤖 Model: Ridge Regression (akurasi ~97%)")
    return komoditas_sel, region_sel, gaji_harian


# ── KOMPONEN UI ──
def render_metrics_row(harga_kini, gaji_harian, row_dict):
    food_budget = gaji_harian * FOOD_SHARE
    ratio = harga_kini / food_budget if food_budget > 0 else 0
    row_copy = dict(row_dict); row_copy["price_to_budget_ratio"] = ratio
    risk_score = hitung_risk_score(row_copy); risk_level = risk_score_to_level(risk_score)
    risk_color = RISK_COLOR.get(risk_level, "#888")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Harga Terkini", f"Rp{harga_kini:,.0f}",
                help="Harga terakhir yang tercatat di pasar.")
    col2.metric("📊 Porsi dari Anggaran Makan", f"{ratio*100:.0f}%",
                delta="⚠️ Di atas batas aman (60%)" if ratio > JONSSON_TOOLE else "✅ Masih aman",
                delta_color="off", help="Berapa persen harga dari anggaran makan harian. Batas aman = 60%.")
    col3.metric("🍱 Anggaran Makan / Hari", f"Rp{food_budget:,.0f}",
                help="55% dari gaji harian — porsi wajar untuk kebutuhan pangan.")
    risk_desc = {"aman":"Harga dalam kisaran wajar","waspada":"Harga perlu dipantau","kritis":"Harga butuh perhatian"}
    with col4:
        st.markdown(
            f"""<div class="risk-badge" style="border-left:4px solid {risk_color};">
                <div class="risk-label">⚠️ Tingkat Risiko Harga</div>
                <div class="risk-value" style="color:{risk_color};">{RISK_LABEL[risk_level]}</div>
                <div class="risk-sub">Skor: {risk_score}/8 &nbsp;·&nbsp; {risk_desc[risk_level]}</div>
            </div>""", unsafe_allow_html=True)
    return risk_score, risk_level

def render_penjelasan_risiko():
    st.markdown("""<div class="info-box">
        <strong>📖 Apa itu Tingkat Risiko Harga?</strong><br>
        Skor risiko (0–8) dihitung otomatis dari 4 faktor:<br>
        &nbsp;&nbsp;• Apakah harga melampaui anggaran makan Anda<br>
        &nbsp;&nbsp;• Seberapa sering harga naik-turun dalam seminggu<br>
        &nbsp;&nbsp;• Tren harga 7 hari terakhir<br>
        &nbsp;&nbsp;• Apakah harga mendekati rekor tertinggi historis<br>
        <span style="color:#16A34A;font-weight:600;">🟢 Aman (0–1)</span> &nbsp;
        <span style="color:#D97706;font-weight:600;">🟡 Waspada (2–5)</span> &nbsp;
        <span style="color:#DC2626;font-weight:600;">🔴 Kritis (6–8)</span>
        </div>""", unsafe_allow_html=True)

def render_chart_historis(df_aff, komoditas, region):
    sub = df_aff[(df_aff["komoditas"]==komoditas)&(df_aff["region"]==region)].copy()
    if sub.empty: st.warning("Data tidak tersedia."); return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["tanggal"], y=sub["harga_rp"], mode="lines",
                             line=dict(color="#6366F1",width=2.5), name="Harga",
                             hovertemplate="<b>%{x|%d %b %Y}</b><br>Rp%{y:,.0f}<extra></extra>"))
    for lvl, clr in RISK_COLOR.items():
        s = sub[sub["risk_level"]==lvl]
        if s.empty: continue
        fig.add_trace(go.Scatter(x=s["tanggal"], y=s["harga_rp"], mode="markers",
                                 marker=dict(color=clr,size=4,opacity=0.7), name=RISK_LABEL[lvl],
                                 hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>Rp%{{y:,.0f}}<br>Risiko: {RISK_LABEL[lvl]}<extra></extra>"))
    fig.update_layout(title=f"📈 Riwayat Harga {KOMODITAS_LABEL[komoditas]} — {region}",
                      xaxis_title="Tanggal", yaxis_title="Harga (Rp)",
                      legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                      height=380, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div class="info-box">
        <strong>📖 Cara membaca grafik riwayat harga:</strong><br>
        &nbsp;&nbsp;🔵 <strong>Garis biru</strong> — pergerakan harga dari hari ke hari.<br>
        &nbsp;&nbsp;🟢 <strong>Titik hijau</strong> — harga level <em>Aman</em>: dalam batas anggaran dan stabil.<br>
        &nbsp;&nbsp;🟡 <strong>Titik kuning</strong> — harga level <em>Waspada</em>: mulai membebani atau menunjukkan tren naik.<br>
        &nbsp;&nbsp;🔴 <strong>Titik merah</strong> — harga level <em>Kritis</em>: melampaui anggaran atau mendekati rekor tertinggi.
        </div>""", unsafe_allow_html=True)

def render_chart_risiko_semua(latest_per_kom, gaji_harian, region_sel):
    if latest_per_kom.empty or "risk_score" not in latest_per_kom.columns:
        st.info("Data risiko tidak tersedia."); return
    df_plot = latest_per_kom.copy()
    def recalc(r):
        rc = dict(r); fp = gaji_harian * FOOD_SHARE
        rc["price_to_budget_ratio"] = r["harga_rp"] / fp if fp > 0 else 0
        return hitung_risk_score(rc)
    df_plot["risk_score"] = df_plot.apply(recalc, axis=1)
    df_plot["risk_level"] = df_plot["risk_score"].apply(risk_score_to_level)
    df_plot["komoditas_label"] = df_plot["komoditas"].map(KOMODITAS_LABEL)
    df_plot["harga_fmt"] = df_plot["harga_rp"].apply(lambda x: f"Rp{x:,.0f}")
    df_plot = df_plot.sort_values("risk_score", ascending=True)
    colors = df_plot["risk_level"].map(RISK_COLOR).tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_plot["komoditas_label"], x=df_plot["risk_score"], orientation="h",
        marker_color=colors,
        text=[f"  {row['risk_score']}/8 — {RISK_LABEL[row['risk_level']]}  ({row['harga_fmt']})"
              for _, row in df_plot.iterrows()],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Skor Risiko: %{x}/8<extra></extra>",
    ))
    fig.add_vline(x=2, line_dash="dot", line_color="#D97706",
                  annotation_text="Waspada", annotation_position="top")
    fig.add_vline(x=6, line_dash="dot", line_color="#DC2626",
                  annotation_text="Kritis", annotation_position="top")
    fig.update_layout(
        title=f"🔍 Tingkat Risiko Harga Terkini — {region_sel}",
        xaxis=dict(title="Skor Risiko (0 = sangat aman, 8 = sangat kritis)", range=[0, 12]),
        yaxis=dict(title=""), height=380,
        margin=dict(l=10, r=10, t=60, b=40), showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(99,102,241,0.12)")
    st.plotly_chart(fig, use_container_width=True)

def render_chart_prediksi_minggu(df_pred, harga_kini, gaji_harian):
    if df_pred is None or df_pred.empty: st.warning("Prediksi tidak tersedia."); return
    batas_aman = gaji_harian * FOOD_SHARE * JONSSON_TOOLE
    today = pd.Timestamp(date.today())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([df_pred["tanggal"],df_pred["tanggal"][::-1]]),
        y=pd.concat([df_pred["upper_rp"],df_pred["lower_rp"][::-1]]),
        fill="toself", fillcolor="rgba(99,102,241,0.10)",
        line=dict(color="rgba(255,255,255,0)"), name="Kisaran Harga (90%)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=df_pred["tanggal"], y=df_pred["pred_rp"], mode="lines+markers",
        line=dict(color="#6366F1",width=2.5), marker=dict(size=9, color="#6366F1"), name="Prediksi Harga",
        hovertemplate="<b>%{x|%A, %d %b %Y}</b><br>Prediksi: Rp%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[today], y=[harga_kini], mode="markers",
        marker=dict(color="#F59E0B",size=14,symbol="diamond"), name="Harga Hari Ini",
        hovertemplate=f"<b>Hari ini</b><br>Rp{harga_kini:,.0f}<extra></extra>"))
    fig.add_hline(y=batas_aman, line_dash="dash", line_color="#DC2626",
                  annotation_text=f"Batas Aman (Rp{batas_aman:,.0f})",
                  annotation_position="top left")
    fig.update_layout(title="🔮 Proyeksi Harga Harian — 7 Hari ke Depan",
                      xaxis_title="Tanggal", yaxis_title="Harga (Rp)",
                      height=400, hovermode="x unified",
                      legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div class="info-box">
        <strong>📖 Cara membaca grafik prediksi harian:</strong><br>
        &nbsp;&nbsp;🔵 <strong>Garis + titik ungu</strong> — perkiraan harga setiap hari dalam 7 hari ke depan.<br>
        &nbsp;&nbsp;💠 <strong>Area ungu muda</strong> — kisaran harga yang paling mungkin terjadi (tingkat kepercayaan 90%).<br>
        &nbsp;&nbsp;🔶 <strong>Titik berlian kuning</strong> — harga terakhir tercatat di pasar (hari ini).<br>
        &nbsp;&nbsp;🔴 <strong>Garis merah putus-putus</strong> — batas harga aman sesuai anggaran makan Anda.
        </div>""", unsafe_allow_html=True)

def render_chart_prediksi_bulan(df_pred, harga_kini, gaji_harian):
    if df_pred is None or df_pred.empty: st.warning("Prediksi tidak tersedia."); return
    batas_aman = gaji_harian * FOOD_SHARE * JONSSON_TOOLE
    today = pd.Timestamp(date.today())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([df_pred["tanggal"],df_pred["tanggal"][::-1]]),
        y=pd.concat([df_pred["upper_rp"],df_pred["lower_rp"][::-1]]),
        fill="toself", fillcolor="rgba(99,102,241,0.10)",
        line=dict(color="rgba(255,255,255,0)"), name="Kisaran Harga (90%)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=df_pred["tanggal"], y=df_pred["pred_rp"], mode="lines+markers",
        line=dict(color="#6366F1",width=2.5), marker=dict(size=12, color="#6366F1"), name="Prediksi Per Minggu",
        text=df_pred["label"],
        hovertemplate="<b>%{text}</b><br>%{x|%d %b %Y}<br>Rp%{y:,.0f}<extra></extra>"))
    for _, row in df_pred.iterrows():
        fig.add_annotation(x=row["tanggal"], y=row["pred_rp"], text=row["label"],
                           showarrow=False, yshift=18,
                           font=dict(size=11,color="#6366F1",family="DM Sans"))
    fig.add_trace(go.Scatter(
        x=[today], y=[harga_kini], mode="markers",
        marker=dict(color="#F59E0B",size=14,symbol="diamond"), name="Harga Hari Ini",
        hovertemplate=f"<b>Hari ini</b><br>Rp{harga_kini:,.0f}<extra></extra>"))
    fig.add_hline(y=batas_aman, line_dash="dash", line_color="#DC2626",
                  annotation_text=f"Batas Aman (Rp{batas_aman:,.0f})",
                  annotation_position="top left")
    fig.update_layout(title="🔮 Proyeksi Harga Per Minggu — 1 Bulan ke Depan",
                      xaxis_title="Tanggal", yaxis_title="Harga (Rp)",
                      height=420, hovermode="x unified",
                      legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div class="info-box">
        <strong>📖 Cara membaca grafik prediksi bulanan:</strong><br>
        &nbsp;&nbsp;🔵 <strong>Titik + label Minggu 1–4</strong> — perkiraan harga di akhir setiap minggu.<br>
        &nbsp;&nbsp;💠 <strong>Area ungu muda</strong> — kisaran harga yang paling mungkin terjadi (90% kepercayaan).<br>
        &nbsp;&nbsp;🔶 <strong>Titik berlian kuning</strong> — harga terakhir tercatat (hari ini sebagai titik awal).<br>
        &nbsp;&nbsp;🔴 <strong>Garis merah putus-putus</strong> — batas harga aman sesuai anggaran makan Anda.
        </div>""", unsafe_allow_html=True)

def render_tabel_prediksi_minggu(df_pred, gaji_harian):
    if df_pred is None or df_pred.empty: return
    rows = []
    for _, r in df_pred.iterrows():
        aff = hitung_affordability(r["pred_rp"], gaji_harian)
        rows.append({"Hari ke-": int(r["horizon"]),
                     "Tanggal": r["tanggal"].strftime("%a, %d %b %Y"),
                     "Prediksi Harga": f"Rp{r['pred_rp']:,.0f}",
                     "Harga Minimum": f"Rp{r['lower_rp']:,.0f}",
                     "Harga Maksimum": f"Rp{r['upper_rp']:,.0f}",
                     "% Anggaran Makan": f"{aff['ratio']*100:.0f}%",
                     "Status": "✅ Terjangkau" if aff["affordability_ok"] else "⚠️ Di atas batas"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def render_tabel_prediksi_bulan(df_pred, gaji_harian):
    if df_pred is None or df_pred.empty: return
    rows = []
    for _, r in df_pred.iterrows():
        aff = hitung_affordability(r["pred_rp"], gaji_harian)
        rows.append({"Periode": r["label"],
                     "Tanggal Perkiraan": r["tanggal"].strftime("%d %b %Y"),
                     "Prediksi Harga": f"Rp{r['pred_rp']:,.0f}",
                     "Harga Minimum": f"Rp{r['lower_rp']:,.0f}",
                     "Harga Maksimum": f"Rp{r['upper_rp']:,.0f}",
                     "% Anggaran Makan": f"{aff['ratio']*100:.0f}%",
                     "Status": "✅ Terjangkau" if aff["affordability_ok"] else "⚠️ Di atas batas"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def render_keterangan_tabel():
    st.markdown("""<div class="info-box" style="margin-top:8px;">
        <strong>📖 Keterangan kolom tabel:</strong><br>
        &nbsp;&nbsp;• <strong>Prediksi Harga</strong> — perkiraan terbaik harga pada periode tersebut.<br>
        &nbsp;&nbsp;• <strong>Harga Minimum / Maksimum</strong> — kisaran harga paling mungkin terjadi (tingkat kepercayaan 90%).<br>
        &nbsp;&nbsp;• <strong>% Anggaran Makan</strong> — persentase harga terhadap anggaran makan harian. Aman selama ≤ 60%.
        </div>""", unsafe_allow_html=True)

def render_evaluasi_model(eval_report):
    st.subheader("🔍 Akurasi Model per Komoditas")
    per_kom = eval_report.get("per_komoditas", [])
    if per_kom:
        df_pk = pd.DataFrame(per_kom)
        df_pk["komoditas"] = df_pk["komoditas"].map(KOMODITAS_LABEL).fillna(df_pk["komoditas"])
        df_pk["MAE"]  = df_pk["MAE"].apply(lambda x: f"Rp{x:,.0f}")
        df_pk["MAPE"] = df_pk["MAPE"].apply(lambda x: f"{x:.2f}%")
        df_pk.rename(columns={"komoditas":"Komoditas","n":"Jumlah Data Uji",
                               "MAE":"Selisih Rata-rata (Rp)","MAPE":"Kesalahan (%)"}, inplace=True)
        st.dataframe(df_pk[["Komoditas","Selisih Rata-rata (Rp)","Kesalahan (%)","Jumlah Data Uji"]],
                     use_container_width=True, hide_index=True)
        st.markdown("""<div class="info-box" style="margin-top:8px;">
            <strong>📖 Cara membaca tabel akurasi:</strong><br>
            &nbsp;&nbsp;• <strong>Selisih Rata-rata (Rp)</strong> — rata-rata selisih antara harga prediksi dan harga nyata.<br>
            &nbsp;&nbsp;• <strong>Kesalahan (%)</strong> — rata-rata persentase meleset dari harga nyata. Semakin kecil = semakin akurat.
            </div>""", unsafe_allow_html=True)


# ── MAIN ──
def main():
    try:
        meta, le_kom, le_reg, multi_models, multi_scalers, conformal_q, eval_report = load_artifacts()
        df_aff, df_feat = load_data()
    except Exception as e:
        st.error(f"Gagal memuat data atau model: {e}"); st.stop()

    tanggal_data_terakhir = df_feat["tanggal"].max()
    komoditas_sel, region_sel, gaji_harian = render_sidebar(tanggal_data_terakhir)

    st.title("Prediksi Harga Pangan DKI Jakarta")
    st.markdown("Pantau harga bahan pokok, lihat proyeksi ke depan, dan dapatkan "
                "**saran belanja berbasis AI** yang disesuaikan dengan kondisi keuangan Anda.")
    st.markdown("---")

    mask = (df_aff["komoditas"]==komoditas_sel) & (df_aff["region"]==region_sel)
    df_last = df_aff[mask].sort_values("tanggal").tail(1).reset_index(drop=True)
    if df_last.empty: st.error("Data tidak ditemukan untuk pilihan ini."); st.stop()

    row_dict   = df_last.to_dict(orient="records")[0]
    harga_kini = float(row_dict["harga_rp"])
    tgl_str    = str(row_dict.get("tanggal",""))[:10]

    food_budget_user = gaji_harian * FOOD_SHARE
    row_for_risk = dict(row_dict)
    row_for_risk["price_to_budget_ratio"] = harga_kini / food_budget_user if food_budget_user > 0 else 0

    tab1, tab2, tab3 = st.tabs(["📊 Kondisi Saat Ini","🔮 Prediksi & Saran","ℹ️ Tentang Aplikasi"])

    with tab1:
        st.subheader(f"🏪 {KOMODITAS_LABEL[komoditas_sel]} · {region_sel}")
        st.caption(f"Data pasar per: **{tgl_str}** · Gaji Anda: **Rp{gaji_harian*30:,.0f}/bulan** · "
                   f"Anggaran makan: **Rp{food_budget_user:,.0f}/hari**")
        risk_score, risk_level = render_metrics_row(harga_kini, gaji_harian, row_dict)
        render_penjelasan_risiko()
        st.markdown("---")
        render_chart_historis(df_aff, komoditas_sel, region_sel)
        st.markdown("---")
        st.subheader(f"📉 Tingkat Risiko Semua Bahan Pokok — {region_sel}")
        latest_per_kom = get_latest_per_komoditas(df_aff, region_sel)
        render_chart_risiko_semua(latest_per_kom, gaji_harian, region_sel)

    with tab2:
        st.subheader(f"Prediksi Harga: {KOMODITAS_LABEL[komoditas_sel]} · {region_sel}")
        mode_prediksi = st.radio("📅 Pilih periode prediksi:", options=["minggu","bulan"],
                                 format_func=lambda x: " 1 Minggu ke Depan (per hari)" if x=="minggu"
                                 else " 1 Bulan ke Depan (per minggu)", horizontal=True)
        st.markdown("---")

        if mode_prediksi == "minggu":
            with st.spinner("⏳ Menghitung prediksi harga harian..."):
                df_pred = predict_mingguan(komoditas_sel, region_sel, df_feat, meta,
                                           multi_models, multi_scalers, conformal_q)
            if df_pred is None: st.warning("Data fitur tidak cukup."); st.stop()
            render_chart_prediksi_minggu(df_pred, harga_kini, gaji_harian)
            with st.expander("📋 Lihat Tabel Detail Prediksi Harian"):
                render_tabel_prediksi_minggu(df_pred, gaji_harian)
                render_keterangan_tabel()
        else:
            with st.spinner("⏳ Menghitung prediksi harga per minggu..."):
                df_pred = predict_bulanan(komoditas_sel, region_sel, df_feat, meta,
                                          multi_models, multi_scalers, conformal_q)
            if df_pred is None: st.warning("Data fitur tidak cukup."); st.stop()
            render_chart_prediksi_bulan(df_pred, harga_kini, gaji_harian)
            with st.expander("📋 Lihat Tabel Detail Prediksi Per Minggu"):
                render_tabel_prediksi_bulan(df_pred, gaji_harian)
                render_keterangan_tabel()

        over_limit = sum(1 for _,r in df_pred.iterrows()
                         if not hitung_affordability(r["pred_rp"], gaji_harian)["affordability_ok"])
        total_pred = len(df_pred)
        if over_limit == 0:
            st.success("✅ **Kabar baik!** Seluruh proyeksi masih dalam batas anggaran makan Anda.")
        elif over_limit < total_pred:
            st.warning(f"⚠️ **Perhatian:** {over_limit} dari {total_pred} periode proyeksi "
                       f"diperkirakan melampaui batas anggaran makan Anda.")
        else:
            st.error("🔴 **Waspada:** Seluruh proyeksi melebihi batas anggaran makan Anda. "
                     "Pertimbangkan alternatif bahan pengganti.")

        st.markdown("---")
        st.subheader("🤖 Analisis & Saran Belanja Cerdas")
        label_mode = "7 hari ke depan" if mode_prediksi == "minggu" else "1 bulan ke depan"
        st.markdown(
            f"Klik tombol di bawah untuk mendapatkan **analisis personal** harga "
            f"**{KOMODITAS_LABEL[komoditas_sel]}** di **{region_sel}** "
            f"untuk **{label_mode}** — disesuaikan dengan kondisi keuangan Anda."
        )
        if st.button("✨ Analisis Sekarang", type="primary", use_container_width=True):
            with st.spinner("⏳ Menyusun analisis personal..."):
                saran = generate_saran_pintar(
                    KOMODITAS_LABEL[komoditas_sel], region_sel,
                    harga_kini, gaji_harian, df_pred,
                    risk_level, risk_score, mode_prediksi, eval_report
                )
            st.markdown("---")
            st.markdown(saran)
            st.markdown("---")

    with tab3:
        st.subheader("ℹ️ Tentang Aplikasi Ini")
        st.markdown("""
        Aplikasi ini membantu masyarakat DKI Jakarta memantau harga 9 bahan pokok strategis,
        memperkirakan harga ke depan, dan membuat keputusan belanja yang lebih hemat.

        ---
        ### 🧮 Bagaimana Cara Kerja Prediksi?

        Prediksi menggunakan **Ridge Regression** yang belajar dari pola harga historis,
        tren mingguan, rata-rata bergerak, dan volatilitas harga.

        - **1 Minggu ke Depan** → prediksi harian (Hari 1–7): gabungan model t+1 (hari 1–3)
          dan t+7 (hari 4–7) secara bertahap.
        - **1 Bulan ke Depan** → prediksi per minggu (Minggu 1–4): model khusus untuk
          horizon 7, 14, 21, dan 30 hari.
        - **Kisaran harga** (area biru muda) menggunakan metode *Conformal Prediction* —
          90% kemungkinan harga nyata jatuh di rentang tersebut.

        ---
        ### 📦 Data & Komoditas

        - **Sumber data**: Harga pasar DKI Jakarta (Jan 2024 – Des 2025)
        - **Komoditas**: Ayam Ras · Bawang Merah · Bawang Putih · Beras Premium ·
          Cabai Merah Keriting · Cabai Rawit Merah · Daging Sapi · Gula Pasir · Telur Ayam
        - **Wilayah**: Jakarta Barat · Pusat · Selatan · Timur · Utara
        """)
        st.markdown("---")
        render_evaluasi_model(eval_report)

if __name__ == "__main__":
    main()