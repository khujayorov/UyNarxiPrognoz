import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("house_model.pkl")

st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 AI House Price Prediction System")
st.markdown("Sun'iy intellekt yordamida uy narxini prognoz qilish")

# ----- Model prediction section -----
st.sidebar.header("Uy parametrlarini kiriting")
area = st.sidebar.slider("Area (m²)", 50, 500, 120)
bedrooms = st.sidebar.slider("Bedrooms", 1, 20, 2)
age = st.sidebar.slider("House Age", 0, 30, 5)
location = st.sidebar.slider("Location Score", 1, 10, 5)

if st.sidebar.button("Predict Price"):
    features = np.array([[area, bedrooms, age, location]])
    prediction = model.predict(features)
    st.subheader("💰 Estimated House Price")
    st.success(f"${prediction[0]:,.0f}")

st.divider()

# ----- Dataset manager (CRUD) -----

DATA_FILE = "house_price_dataset_1200.csv"
EXPECTED_COLS = ["area", "bedrooms", "age", "location", "price"]

if "house_df" not in st.session_state:
    try:
        df = pd.read_csv(DATA_FILE)
        for c in EXPECTED_COLS:
            if c not in df.columns:
                df[c] = 0
        st.session_state.house_df = df[EXPECTED_COLS].copy()
    except FileNotFoundError:
        st.session_state.house_df = pd.DataFrame(columns=EXPECTED_COLS)

st.subheader("🗂️ Uy ma'lumotlarini boshqarish")

with st.expander("Yangi yozuv qo'shish", expanded=True):
    add_area = st.number_input("Area (m²)", min_value=10, max_value=1000, value=120)
    add_bedrooms = st.number_input("Bedrooms", min_value=1, max_value=20, value=2)
    add_age = st.number_input("House Age", min_value=0, max_value=100, value=5)
    add_location = st.number_input("Location Score", min_value=1, max_value=10, value=5)
    add_price = st.number_input("Real house price", min_value=1000.0, max_value=10000000.0, value=150000.0, step=1000.0)

    if st.button("Yozuv qo'shish"):
        new_row = {
            "area": add_area,
            "bedrooms": add_bedrooms,
            "age": add_age,
            "location": add_location,
            "price": add_price,
        }
        st.session_state.house_df = pd.concat([st.session_state.house_df, pd.DataFrame([new_row])], ignore_index=True)
        st.success("🎉 Yangi yozuv qo'shildi")

st.write("### 📋 Mavjud yozuvlar")
edited_df = st.data_editor(
    st.session_state.house_df,
    num_rows="dynamic",
    use_container_width=True,
)

if st.button("O'zgartirishlarni saqlash"):
    st.session_state.house_df = edited_df
    st.success("✅ O'zgartirishlar saqlandi")

# idx tanlangan qatorlarni o’chirish imkoniyati
selected_rows = st.multiselect("O'chirish uchun satrni tanlang (index bo'yicha)", options=st.session_state.house_df.index.tolist())
if st.button("Tanlangan satrlarni o'chirish"):
    if len(selected_rows) > 0:
        st.session_state.house_df = st.session_state.house_df.drop(selected_rows).reset_index(drop=True)
        st.success(f"🗑️ {len(selected_rows)} satr o'chirildi")
    else:
        st.warning("Iltimos, avval satrni tanlang")

# Saqlash imkoniyati
if st.sidebar.button("Joriy datasetni saqlash"):
    st.session_state.house_df.to_csv(DATA_FILE, index=False)
    st.success(f"🗄️ '{DATA_FILE}' ga saqlandi")

st.divider()

# ----- Beautiful images -----
st.subheader(" 🏠 Uy ma'lumotlar: Zamonaviy architektura asosida qurilgan uylar")
image_urls = [
    "https://images.unsplash.com/photo-1564013799919-ab600027ffc6?auto=format&fit=crop&w=400&h=300&q=80",
    "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=400&h=300&q=80",
    "https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?auto=format&fit=crop&w=400&h=300&q=80",
    "https://images.unsplash.com/photo-1552321554-5fefe8c9ef14?auto=format&fit=crop&w=400&h=300&q=80",
    "https://images.unsplash.com/photo-1522708323590-d24dbb6b0267?auto=format&fit=crop&w=400&h=300&q=80",
    "https://images.unsplash.com/photo-1568605114967-8130f3a36994?auto=format&fit=crop&w=400&h=300&q=80",
]
cols = st.columns(3)
for idx, url in enumerate(image_urls):
    cols[idx % 3].image(url, use_container_width=True, caption=f"Uyning tashqi va ichki ko'rinishi {idx+1}")

st.info("📌 Jadvaldan yozuvlar qo'shish, tahrirlash va o'chirish mumkin. Predict tugmasi model yordamida baho beradi.")
