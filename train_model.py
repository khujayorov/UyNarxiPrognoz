# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# import joblib

# data = pd.read_csv("house_price_dataset_1200.csv")

# X = data.drop("price", axis=1)
# y = data["price"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# model = LinearRegression()
# model.fit(X_train, y_train)

# pred = model.predict(X_test)

# accuracy = r2_score(y_test, pred)

# print("Model Accuracy:", accuracy)

# joblib.dump(model, "house_model.pkl")

import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# Streamlit interfeysini sozlash
st.set_page_config(page_title="Uy narxini bashorat", page_icon="🏠", layout="wide")

# Minimal styling
st.markdown(
    """
    <style>
    /* Animated background gradient */
    .stApp {
      background: linear-gradient(135deg, #0b3d91 0%, #73b3ff 35%, #e0f2ff 100%);
      background-size: 400% 400%;
      animation: gradientBG 18s ease infinite;
      color: #0d1d3a;
    }

    @keyframes gradientBG {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }

    /* Heading contrast */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
      color: #0b1d3c;
      text-shadow: 1px 1px 6px rgba(0, 0, 0, 0.16);
    }

    /* All text should be readable on gradient */
    .stApp, .stApp * {
      color: #0b1d3c !important;
    }

    /* Inputs and form elements */
    input, textarea, select, button, .stTextInput, .stNumberInput {
      background: rgba(255, 255, 255, 0.12) !important;
      color: #0b1d3c !important;
      border: 1px solid rgba(255, 255, 255, 0.35) !important;
    }

    /* Button styling */
    .stButton > button {
      background-color: #0066cc;
      color: white;
      border-radius: 12px;
      padding: 10px 14px;
      border: 1px solid rgba(255, 255, 255, 0.25);
      box-shadow: 0 8px 18px rgba(0, 0, 0, 0.12);
    }

    .stButton > button:hover {
      background-color: #004a9f;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
      background: rgba(255, 255, 255, 0.88);
      border-radius: 16px;
      padding: 18px 14px 14px 14px;
      backdrop-filter: blur(14px);
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.14);
    }

    /* File uploader styling */
    section[data-testid="stSidebar"] .stFileUploader {
      background: rgba(20, 30, 50, 0.45) !important;
      border: 1px dashed rgba(255, 255, 255, 0.45) !important;
      border-radius: 14px !important;
      padding: 14px !important;
      color: #f4f7ff !important;
    }
    section[data-testid="stSidebar"] .stFileUploader label {
      color: #f0f5ff !important;
    }
    section[data-testid="stSidebar"] .stFileUploader button {
      background: rgba(0, 102, 204, 0.95) !important;
    }

    /* Table header contrast */
    .stDataFrame thead tr th {
      background: rgba(10, 32, 60, 0.75) !important;
      color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏠 Uy narxini bashorat qilish")
st.markdown(
    "Bu ilova yuklangan CSV ma'lumotlariga asoslanib uy narxini bashorat qiladi.\n"
    "Faylni yuklashdan oldin ustunlar sozlanganligiga e'tibor bering: oxirgi ustun maqsad (price) bo'lishi kerak."
)

# Model va boshqa obyektlar uchun fayl nomlari
MODEL_FILE = 'regression_model.pkl'
SCALER_FILE = 'scaler.pkl'
IMPUTER_FILE = 'imputer.pkl'
META_FILE = 'model_meta.json'
SAMPLE_CSV = 'house_price_dataset_1200.csv'

# CSV faylni yuklash
st.sidebar.header("1) Ma'lumotni tanlang")
uploaded_file = st.sidebar.file_uploader("CSV faylni yuklang", type=["csv"])
use_sample = st.sidebar.checkbox("Namuna (sample) ma'lumot bilan ishlash", value=False)

# Agar foydalanuvchi sample ni tanlasa, upload qilinmagan deb hisoblanadi va lokal fayldan o'qiladi
if use_sample:
    uploaded_file = None

# Model va boshqa obyektlarni yuklash yoki o'qitish
if uploaded_file is not None or use_sample:
    # CSV faylni o'qish va xatoliklarni boshqarish
    try:
        # Fayl manbaini aniqlash
        if use_sample:
            if not os.path.exists(SAMPLE_CSV):
                st.error(f"Namuna fayl topilmadi: {SAMPLE_CSV}")
                st.stop()
            df = pd.read_csv(SAMPLE_CSV)
            st.info("Namuna ma'lumotlar (sample) yuklandi.")
        else:
            df = pd.read_csv(uploaded_file)

        if df.empty or len(df.columns) < 2:
            st.error("Faylda ma'lumotlar yetarli emas yoki noto'g'ri formatda!")
            st.stop()

        with st.expander("Yuklangan ma'lumotlarni ko'rish"):
            st.dataframe(df)

        with st.expander("Ma'lumotlar statistikasi"):
            st.write(df.describe())

    except Exception as e:
        st.error(f"Faylni o'qishda xatolik: {e}")
        st.stop()

    # Model va boshqa obyektlarni fayldan yuklash yoki yangidan o'qitish
    meta = None
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:
            meta = None

    saved_model_ok = (
        os.path.exists(MODEL_FILE)
        and os.path.exists(SCALER_FILE)
        and os.path.exists(IMPUTER_FILE)
        and meta is not None
        and meta.get('columns') == list(df.columns)
    )

    if saved_model_ok:
        # Mavjud modelni yuklash
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        imputer = joblib.load(IMPUTER_FILE)
        label_encoders = {}
        for col in df.columns[:-1]:
            if df[col].dtype == "object":
                le_file = f'label_encoder_{col}.pkl'
                if os.path.exists(le_file):
                    label_encoders[col] = joblib.load(le_file)
        st.success("Saqlangan model va obyektlar yuklandi.")
    else:
        # Kategorik ma'lumotlarni kodlash
        label_encoders = {}
        for column in df.columns[:-1]:
            if df[column].dtype == "object":
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                label_encoders[column] = le

        # NaN qiymatlarni to'ldirish
        imputer = SimpleImputer(strategy='mean')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
        # Feature va target ustunlarini ajratish
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Ma'lumotlarni standartlashtirish
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Ma'lumotlarni o'quv va test qismga ajratish
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Multiple regression modelini o'qitish
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Model va boshqa obyektlarni saqlash
        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        joblib.dump(imputer, IMPUTER_FILE)
        for col, le in label_encoders.items():
            joblib.dump(le, f'label_encoder_{col}.pkl')

        # Saqlash uchun meta ma'lumotlar
        meta_data = {
            'columns': list(df.columns),
            'target': df.columns[-1],
        }
        with open(META_FILE, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)

        st.success("Model va obyektlar saqlandi va o'qitildi.")

    # Model aniqligini ko'rsatish (agar test ma'lumotlari mavjud bo'lsa)
    if 'X_test' in locals() and 'y_test' in locals():
        score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        col1, col2, col3 = st.columns(3)
        col1.metric("R^2 (aniqlik)", f"{score:.2f}")
        col2.metric("MAE", f"{mae:.2f}")
        col3.metric("MSE", f"{mse:.2f}")

        with st.expander("Test natijalarini grafik ko'rish"):
            chart_data = pd.DataFrame({
                "Haqiqiy": y_test,
                "Bashorat": y_pred,
            })
            st.line_chart(chart_data)

    # Foydalanuvchi kiritgan qiymatlarni olish
    st.sidebar.header("2) Kiruvchi ma'lumotlar")
    user_inputs = []
    for i, column in enumerate(df.columns[:-1]):
        if column in label_encoders:
            options = label_encoders[column].classes_
            user_input = st.sidebar.selectbox(f"{column} ni tanlang:", options)
            user_input = label_encoders[column].transform([user_input])[0]
        else:
            user_input = st.sidebar.number_input(
                f"{column} qiymatini kiriting:",
                min_value=float(df[column].min()),
                max_value=float(df[column].max()),
                value=float(df[column].mean())
            )
        user_inputs.append(user_input)

    # Bashorat qilish
    if st.sidebar.button("Bashorat qilish"):
        try:
            user_inputs_scaled = scaler.transform([user_inputs])
            prediction = model.predict(user_inputs_scaled)
            st.subheader("Bashorat natijasi:")
            st.write(f"Bashorat qilingan qiymat: {prediction[0]:.2f}")

            # Natijani jadvalga chiqarish va yuklab olish
            result_df = pd.DataFrame({
                **{col: [val] for col, val in zip(df.columns[:-1], user_inputs)},
                df.columns[-1]: [prediction[0]],
            })
            st.write("Bashorat natijasi jadvali:")
            st.dataframe(result_df)

            csv_bytes = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Bashorat natijasini CSVga yuklab olish",
                data=csv_bytes,
                file_name="bashorat_natijasi.csv",
                mime="text/csv",
            )

            st.info("Agar ilova telefonda ochilgan bo'lsa, bu faylni kompyuterga yuborib ochishingiz mumkin.")
        except Exception as e:
            st.error(f"Bashorat qilishda xatolik: {e}")