# 🚕 PREDIKSI PERMINTAAN TAKSI NYC (FINAL STREAMLIT)
# ==============================================

# 1️⃣ Import Library
# ==============================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# 2️⃣ Load Model & Dataset
# ==============================================
MODEL_PATH = "model_rf_compressed.joblib"
DATA_PATH = "taxi_tripdata.csv"

# Cek file model
if not os.path.exists(MODEL_PATH):
    st.error("❌ File model_rf_compressed.joblib tidak ditemukan. Pastikan sudah disimpan di folder yang sama!")
    st.stop()

# Muat model
model = joblib.load(MODEL_PATH)

# Cek file dataset
if not os.path.exists(DATA_PATH):
    st.error("❌ File taxi_tripdata.csv tidak ditemukan di folder proyek!")
    st.stop()

# Muat dataset
df = pd.read_csv(DATA_PATH)

# Pastikan kolom waktu tersedia
if "pickup_hour" not in df.columns or "pickup_dayofweek" not in df.columns:
    if any("pickup" in c and ("date" in c or "time" in c) for c in df.columns):
        pickup_col = next(c for c in df.columns if "pickup" in c and ("date" in c or "time" in c))
        df[pickup_col] = pd.to_datetime(df[pickup_col], errors="coerce")
        df["pickup_hour"] = df[pickup_col].dt.hour
        df["pickup_dayofweek"] = df[pickup_col].dt.dayofweek
    else:
        st.error("Dataset tidak memiliki kolom waktu pickup_datetime.")
        st.stop()

# 3️⃣ Konfigurasi Halaman
# ==============================================
st.set_page_config(page_title="Prediksi Permintaan Taksi NYC", page_icon="🚕", layout="centered")

st.title("🚕 Prediksi Permintaan Taksi NYC")
st.markdown(
    """
    Aplikasi ini memprediksi **jumlah perjalanan taksi (trip)** berdasarkan:
    - Lokasi penjemputan (**PUlocationID**)  
    - Jam penjemputan (**pickup_hour**)  
    - Hari dalam minggu (**pickup_dayofweek**)  

    ---
    """
)

# 4️⃣ Input Pengguna
# ==============================================
st.subheader("🎯 Masukkan Parameter Prediksi")

col1, col2, col3 = st.columns(3)

with col1:
    pulocation = st.number_input("Lokasi Penjemputan (PUlocationID)", min_value=1, max_value=300, value=69)

with col2:
    pickup_hour = st.slider("Jam Penjemputan (0–23)", 0, 23, 12)

with col3:
    pickup_dayofweek = st.slider("Hari Penjemputan (0=Senin, 6=Minggu)", 0, 6, 3)

# 5️⃣ Prediksi Jumlah Trip
# ==============================================
if st.button("🔍 Prediksi Jumlah Trip"):
    input_data = pd.DataFrame({
        "pulocationid": [pulocation],
        "pickup_hour": [pickup_hour],
        "pickup_dayofweek": [pickup_dayofweek]
    })

    y_pred = model.predict(input_data)[0]
    st.success(f"🚖 **Prediksi Jumlah Trip:** {y_pred:.2f} perjalanan")

# 6️⃣ Analisis Data (EDA)
# ==============================================
st.markdown("---")
st.subheader("📊 Analisis Data Trip (EDA)")

if st.checkbox("Tampilkan Distribusi & Heatmap", value=True):

    # Distribusi Trip per Jam
    st.markdown("### ⏰ Distribusi Trip per Jam")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x="pickup_hour", data=df, color="skyblue", ax=ax)
    ax.set_title("Distribusi Permintaan Trip per Jam", fontsize=12)
    ax.set_xlabel("Jam Penjemputan (0–23)")
    ax.set_ylabel("Jumlah Trip")
    st.pyplot(fig)

    # Distribusi Trip per Hari
    st.markdown("### 📅 Distribusi Trip per Hari (0=Senin ... 6=Minggu)")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x="pickup_dayofweek", data=df, color="orange", ax=ax)
    ax.set_title("Distribusi Permintaan Trip per Hari", fontsize=12)
    ax.set_xlabel("Hari dalam Minggu")
    ax.set_ylabel("Jumlah Trip")
    st.pyplot(fig)

    # Heatmap Jam vs Hari
    st.markdown("### 🔥 Heatmap Permintaan Trip (Jam vs Hari)")
    heat_data = df.groupby(["pickup_dayofweek", "pickup_hour"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heat_data, cmap="YlOrRd", linewidths=0.5, ax=ax)
    ax.set_title("Pola Permintaan Taksi Berdasarkan Hari dan Jam", fontsize=12)
    ax.set_xlabel("Jam Penjemputan (0–23)")
    ax.set_ylabel("Hari (0=Senin ... 6=Minggu)")
    st.pyplot(fig)

else:
    st.info("Centang kotak di atas untuk menampilkan visualisasi EDA.")

# 7️⃣ Footer
# ==============================================
st.markdown("---")
st.caption("Developed by **Hazril N.A. | Final Project Data Analyst 2025** 🚀")
