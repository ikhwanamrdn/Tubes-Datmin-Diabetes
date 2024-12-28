import streamlit as st

def main():
    # Set the title of the app
    st.title("Kelompok 3\nPrediksi dan Clustering Dataset Diabetes")

    # Sidebar for navigation
    st.sidebar.title("Menu Utama")
    page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Prediksi", "Clustering"])

    # Main page with an introduction
    if page == "Beranda":
        st.header("Anggota Kelompok")

        # Display team photos in a row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.image("assets\pikhwan.jpg", caption="Ikhwan Amiruddin\n1202223229")
        with col2:
            st.image("assets\prafif.jpg", caption="Rafif Dzaky Daniswara\n1202223211")
        with col3:
            st.image("assets\palfonsus.jpg", caption="Alfonsus Raditya D Y\n1202223363")
        with col4:
            st.image("assets\ppanji.jpg", caption="Pangerso Panji Birowo\n1202223087")

    # Page for prediction
    elif page == "Prediksi":
        st.header("Prediksi Kemungkinan Diabetes")
        from predict import run as predict_run
        predict_run()

    # Page for clustering
    elif page == "Clustering":
        st.header("Dashboard Hasil Clustering")
        from cluster import run as cluster_run
        cluster_run()

if __name__ == "__main__":
    main()