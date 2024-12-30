import streamlit as st
import numpy as np
import joblib

def run():
    # Load the trained Logistic Regression model with error handling
    try:
        model = joblib.load('log_reg_model.pkl')
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return

    # Input form for prediction
    st.write("Masukkan informasi berikut untuk memprediksi kemungkinan diabetes:")

    # Input fields with validation for numeric inputs
    kehamilan = st.number_input("Jumlah Kehamilan:", min_value=0, step=1)
    glukosa = st.number_input("Glukosa (mg/dL):", min_value=0)
    tekanan_darah = st.number_input("Tekanan Darah (mmHg):", min_value=0)
    ketebalan_kulit = st.number_input("Ketebalan Kulit (mm):", min_value=0)
    insulin = st.number_input("Insulin:", min_value=0)
    bmi = st.number_input("Indeks Massa Tubuh:", min_value=0.0, format="%.2f")
    persen_diabet = st.number_input("Persentase Diabetes Faktor Keturunan:", min_value=0.0, format="%.2f")
    usia = st.number_input("Usia:", min_value=0, step=1)

    # Prediction section
    if st.button("Prediksi"):
        # Create input array
        input_data = np.array([[kehamilan, glukosa, tekanan_darah, ketebalan_kulit, insulin, bmi, persen_diabet, usia]])

        # Make prediction using the model
        prediction = model.predict(input_data)[0]

        # Display the prediction result
        if prediction == 1:
            st.success("Hasil Prediksi: Diabetes positif.")
        else:
            st.success("Hasil Prediksi: Diabetes negatif.")

    # Footer section
    st.markdown("---")
    st.markdown("Model ini menggunakan Logistic Regression dengan data dari dataset Diabetes.")
