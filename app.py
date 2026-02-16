import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Porter Delivery Prediction",
    page_icon="🚚",
    layout="centered"
)

# ======================
# CUSTOM CSS (Fix Container & Card)
# ======================
st.markdown("""
<style>
    /* Mengubah background body */
    .stApp {
        background-color: #f8fafc;
    }

    /* Styling Form sebagai Card */
    [data-testid="stForm"] {
        background: white;
        padding: 40px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    }

    /* Title Styling */
    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        color: #64748b;
        margin-bottom: 30px;
    }

    /* Section Title */
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #2563eb;
        margin-top: 20px;
        margin-bottom: 10px;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 5px;
    }

    /* Button Customization */
    .stButton > button {
        width: 100%;
        background-color: #2563eb !important;
        color: white !important;
        border-radius: 8px !important;
        height: 50px;
        font-weight: 600;
        font-size: 16px;
        border: none;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODEL & METADATA
# ======================
@st.cache_resource
def load_assets():
    # Pastikan nama file sesuai dengan yang Anda simpan
    model = joblib.load("model_final_project.pkl")
    feature_names = joblib.load("final_features.pkl")
    return model, feature_names

try:
    model, feature_names = load_assets()
except Exception as e:
    st.error(f"Gagal memuat file model/fitur. Pastikan 'model_final_project.pkl' dan 'final_features.pkl' tersedia. Error: {e}")
    st.stop()

# ======================
# HEADER
# ======================
st.markdown("<div class='main-title'>🚚 Porter Delivery Time Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Estimasi waktu pengantaran menggunakan Machine Learning</div>", unsafe_allow_html=True)

# ======================
# MAIN FORM (The Container)
# ======================
with st.form("delivery_form"):
    
    # --- SECTION 1: Store & Market ---
    st.markdown("<div class='section-header'>🧾 Informasi Toko & Pasar</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        market_id = st.selectbox("Market ID", [1, 2, 3, 4, 5, 6])
    with col_b:
        order_protocol = st.selectbox("Order Protocol", [1, 2, 3, 4, 5, 6, 7])
    
    store_cats = sorted([
        col.replace("store_primary_category_", "")
        for col in feature_names if col.startswith("store_primary_category_")
    ])
    store_primary_category = st.selectbox("Kategori Toko", store_cats if store_cats else ["Unknown"])

    # --- SECTION 2: Order Details ---
    st.markdown("<div class='section-header'>🛒 Detail Pesanan</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        total_items = st.number_input("Total Items", 1, 50, 4)
    with c2:
        num_distinct_items = st.number_input("Distinct Items", 1, 20, 3)
    with c3:
        subtotal = st.number_input("Subtotal", 0, 50000, 2200)
    
    c4, c5 = st.columns(2)
    with c4:
        min_item_price = st.number_input("Min Item Price", 0, 50000, 500)
    with c5:
        max_item_price = st.number_input("Max Item Price", 0, 50000, 900)

    # --- SECTION 3: System Load ---
    st.markdown("<div class='section-header'>🚦 Kondisi Sistem</div>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        total_onshift_partners = st.number_input("Onshift Partners", 0, 300, 30)
    with s2:
        total_busy_partners = st.number_input("Busy Partners", 0, 300, 25)
    with s3:
        total_outstanding_orders = st.number_input("Outstanding Orders", 0, 500, 60)

    # --- SECTION 4: Time ---
    st.markdown("<div class='section-header'>⏰ Waktu Pemesanan</div>", unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    with t1:
        order_date = st.date_input("Tanggal", datetime.date(2015, 2, 6))
    with t2:
        order_time = st.time_input("Jam", datetime.time(22, 24))

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button("🔮 Prediksi Sekarang")

# ======================
# PREDICTION LOGIC
# ======================
if submit_button:
    # 1. Gabungkan Date & Time
    dt_combined = datetime.datetime.combine(order_date, order_time)
    
    # 2. Siapkan DataFrame Dasar
    new_data = pd.DataFrame([{
        'market_id': market_id,
        'order_protocol': order_protocol,
        'store_primary_category': store_primary_category,
        'total_items': total_items,
        'subtotal': subtotal,
        'num_distinct_items': num_distinct_items,
        'min_item_price': min_item_price,
        'max_item_price': max_item_price,
        'total_onshift_partners': total_onshift_partners,
        'total_busy_partners': total_busy_partners,
        'total_outstanding_orders': total_outstanding_orders
    }])

    # 3. Feature Engineering (Sesuai Notebook)
    new_data['order_hour'] = dt_combined.hour
    new_data['day_of_week'] = dt_combined.weekday()
    new_data['is_weekend'] = 1 if dt_combined.weekday() >= 5 else 0

    # Kalkulasi Rasio (+1 untuk menghindari division by zero)
    new_data['load_ratio'] = new_data['total_outstanding_orders'] / (new_data['total_onshift_partners'] + 1)
    new_data['busy_partners_ratio'] = new_data['total_busy_partners'] / (new_data['total_onshift_partners'] + 1)
    new_data['item_complexity'] = new_data['num_distinct_items'] / (new_data['total_items'] + 1)
    new_data['rush_load'] = new_data['load_ratio'] * new_data['is_weekend']

    # 4. One-Hot Encoding
    # Pastikan kolom market_id dan order_protocol diubah ke string jika saat training di-encode
    new_data['market_id'] = new_data['market_id'].astype(str)
    new_data['order_protocol'] = new_data['order_protocol'].astype(str)

    new_data_encoded = pd.get_dummies(new_data, columns=['market_id', 'order_protocol', 'store_primary_category'], drop_first=True)

    # 5. Reindex agar urutan & jumlah kolom (93 fitur) sama dengan saat training
    new_data_encoded = new_data_encoded.reindex(columns=feature_names, fill_value=0)

    # 6. Prediksi
    prediction = model.predict(new_data_encoded)[0]

    # Hasil
    st.markdown("---")
    st.balloons()
    st.success(f"### ⏱️ Estimasi Waktu Pengantaran: **{prediction:.2f} Menit**")