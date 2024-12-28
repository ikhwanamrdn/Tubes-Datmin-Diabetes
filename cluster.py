import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import joblib

def run():
    # Load Data
    df = pd.read_csv("diabetes.csv")

    # Data Cleaning
    df = df.rename(columns={
        'Pregnancies': 'kehamilan',
        'Glucose': 'glukosa',
        'SkinThickness': 'TebalKulit',
        'DiabetesPedigreeFunction': 'PersenDiabet',
        'Age': 'usia',
        'Outcome': 'hasil'
    })

    # Balance Data
    df_0 = df[df['hasil'] == 0]
    df_1 = df[df['hasil'] == 1]
    df_0_balanced = df_0.sample(n=len(df_1), random_state=42)
    df_balanced = pd.concat([df_0_balanced, df_1])
    df = df_balanced

    # Handle Outliers
    def handle_outliers_iqr(data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = np.where(data < lower_bound, lower_bound, data)
        data = np.where(data > upper_bound, upper_bound, data)
        return data

    for column in df.select_dtypes(include=np.number).columns:
        df[column] = handle_outliers_iqr(df[column])

    # Clustering
    kolom_clustering = ['glukosa', 'BMI', 'usia', 'PersenDiabet', 'kehamilan']
    X = df[kolom_clustering]
    agg_clustering = AgglomerativeClustering(n_clusters=4)
    df['cluster'] = agg_clustering.fit_predict(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Save PCA model
    joblib.dump(pca, 'pca_model.pkl')

    # Streamlit Dashboard
    st.title("Dashboard Hasil Clustering")

    # Tabel Cluster
    st.subheader("Tabel Cluster")
    cluster_summary = df.groupby('cluster').size().reset_index(name='Jumlah Data')
    st.table(cluster_summary)

    # Pie Chart
    st.subheader("Distribusi Cluster")
    cluster_counts = df['cluster'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", len(cluster_counts)))
    ax.axis('equal')
    st.pyplot(fig)

    # Deskripsi Cluster
    st.subheader("Analisis dengan Mengambil 25 Sampel Secara Acak")
    cluster_descriptions = {
        0: "Cluster ini memiliki variasi besar dalam jumlah kehamilan, glukosa, dan usia. Nilai BMI cenderung tinggi. Sebagian besar tidak menderita diabetes.",
        1: "Cluster ini memiliki nilai glukosa dan tekanan darah yang tinggi. BMI bervariasi tetapi cenderung tinggi. Sebagian besar menderita diabetes.",
        2: "Cluster ini memiliki nilai glukosa dan insulin tinggi, serta variasi besar dalam kehamilan dan usia. Sebagian besar tidak menderita diabetes.",
        3: "Cluster ini memiliki glukosa, tekanan darah, dan BMI tinggi. Usia rata-rata lebih tua. Sebagian besar menderita diabetes."
    }

    for cluster, description in cluster_descriptions.items():
        st.markdown(f"### Cluster {cluster}")
        st.markdown(description)

    # Scatterplot PCA
    st.subheader("Visualisasi Clustering dengan Agglomerative Clustering")
    fig, ax = plt.subplots()
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['cluster'], palette='viridis', ax=ax)
    ax.set_title("Hasil Clustering dengan PCA")
    ax.set_xlabel("Komponen Utama 1")
    ax.set_ylabel("Komponen Utama 2")
    st.pyplot(fig)
