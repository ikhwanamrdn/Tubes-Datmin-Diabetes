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
        'SkinThickness': 'tebal_kulit',
        'DiabetesPedigreeFunction': 'persen_diabet',
        'Age': 'usia',
        'Outcome': 'hasil'
    })

    # Balance Data
    df_0 = df[df['hasil'] == 0]
    df_1 = df[df['hasil'] == 1]
    df_0_balanced = df_0.sample(n=len(df_1), random_state=42)
    df_balanced = pd.concat([df_0_balanced, df_1])
    df = df_balanced.reset_index(drop=True)

    # Handle Outliers
    def handle_outliers_iqr(data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = np.clip(data, lower_bound, upper_bound)
        return data

    for column in df.select_dtypes(include=np.number).columns:
        df[column] = handle_outliers_iqr(df[column])

    # Clustering
    kolom_clustering = ['glukosa', 'BMI', 'usia', 'persen_diabet', 'kehamilan']
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

    # Input untuk Memilih Fitur
    selected_feature = st.selectbox("Pilih Fitur untuk Distribusi per Cluster", options=kolom_clustering)

    # Menentukan rentang untuk fitur yang dipilih
    min_value = df[selected_feature].min()
    max_value = df[selected_feature].max()
    selected_range = st.slider(f"Rentang {selected_feature.capitalize()}", min_value=min_value, max_value=max_value, value=(min_value, max_value))

    # Memfilter data berdasarkan rentang yang dipilih
    filtered_data = df[(df[selected_feature] >= selected_range[0]) & (df[selected_feature] <= selected_range[1])]

    # Menampilkan jumlah data yang terfilter
    st.write(f"Jumlah data dalam rentang {selected_range[0]} hingga {selected_range[1]}: {len(filtered_data)}")

    # Tabel distribusi data terhadap cluster
    distribution_table = filtered_data.groupby('cluster').size().reset_index(name='Jumlah Data')
    st.subheader("Distribusi Data terhadap Cluster")
    st.table(distribution_table)

    # Pie Chart untuk Distribusi Fitur
    st.subheader(f"Distribusi {selected_feature.capitalize()} per Cluster")
    feature_distribution = filtered_data.groupby('cluster')[selected_feature].mean()
    fig, ax = plt.subplots()
    ax.pie(
        feature_distribution,
        labels=feature_distribution.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("viridis", len(feature_distribution))
    )
    ax.axis('equal')
    st.pyplot(fig)

    # Distribusi Hasil per Cluster
    st.subheader("Distribusi Hasil per Cluster")
    hasil_distribution = df.groupby(['cluster', 'hasil']).size().unstack(fill_value=0)
    st.table(hasil_distribution)

    # Pie Chart untuk Distribusi Hasil
    st.subheader("Distribusi Hasil per Cluster dalam Bentuk Pie Chart")
    for cluster in hasil_distribution.index:
        st.markdown(f"### Cluster {cluster}")
        fig, ax = plt.subplots()
        ax.pie(
            hasil_distribution.loc [cluster],
            labels=hasil_distribution.columns,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("pastel", len(hasil_distribution.columns))
        )
        ax.axis('equal')
        st.pyplot(fig)

    # Deskripsi Cluster
    st.subheader("Hasil Analisis Cluster")
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
    st.subheader("Visualisasi Clustering dengan PCA Agglomerative Clustering")
    fig, ax = plt.subplots()
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['cluster'], palette='viridis', ax=ax)
    ax.set_title("Hasil Clustering dengan PCA Agglomerative Clustering")
    ax.set_xlabel("Komponen Utama 1")
    ax.set_ylabel("Komponen Utama 2")
    st.pyplot(fig)

if __name__ == "__main__":
    run()