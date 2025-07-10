import streamlit as st
import pandas as pd
import joblib

# ===== Load model dan daftar kolom fitur =====
model = joblib.load('model_rf.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# ===== Judul =====
st.title("Prediksi High Performer Karyawan")

# ===== Form input =====
def user_input():
    Age = st.slider("Umur", 20, 65, 30)
    Salary = st.number_input("Gaji (Rp)", value=5000000)
    Absences = st.number_input("Jumlah Ketidakhadiran", value=0)
    EngagementSurvey = st.slider("Survei Keterlibatan (0-5)", 0.0, 5.0, 3.0, step=0.1)
    SpecialProjectsCount = st.selectbox("Jumlah Proyek Khusus", [0, 1, 2, 3, 4])
    DaysLateLast30 = st.number_input("Jumlah Telat 30 Hari Terakhir", value=0)
    GenderID = st.selectbox("Jenis Kelamin", [0, 1])  # 0 = Pria, 1 = Wanita

    data = {
        'Age': Age,
        'Salary': Salary,
        'Absences': Absences,
        'EngagementSurvey': EngagementSurvey,
        'SpecialProjectsCount': SpecialProjectsCount,
        'DaysLateLast30': DaysLateLast30,
        'GenderID': GenderID
    }

    return pd.DataFrame([data])

input_df = user_input()

# ===== Prediksi =====
if st.button("Prediksi"):
    # Reindex agar sesuai dengan kolom model training
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Hasil Prediksi")
    if pred == 1:
        st.success(f"Karyawan diprediksi sebagai **High Performer** (Probabilitas: {prob:.2f})")
    else:
        st.warning(f"Karyawan diprediksi **Bukan High Performer** (Probabilitas: {prob:.2f})")
