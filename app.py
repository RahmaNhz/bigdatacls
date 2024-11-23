import pandas as pd
import geopandas as gpd
from shapely import wkt
import folium
from folium import Tooltip
from streamlit_folium import st_folium
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# 1. Input Data
data = pd.read_csv('datapadi.csv')

# Group data by Kab/Kota and calculate the mean
data_avg = data.groupby('Kab/Kota').mean().reset_index()

# Normalisasi data
minmax_scaler = MinMaxScaler()
Normalisasi = minmax_scaler.fit_transform(data_avg[['Luas Panen', 'Produktivitas', 'Produksi']])

# Clustering dengan KMeans
k = 4
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
data_avg['Cluster'] = kmeans.fit_predict(Normalisasi)

# Membaca data CSV yang berisi geometri
datapeta = pd.read_csv('jawa_timur.csv')
datapeta['geometry'] = datapeta['geometry'].apply(wkt.loads)

# Membuat GeoDataFrame
gdf = gpd.GeoDataFrame(datapeta, geometry='geometry')

# Gabungkan data_avg dengan gdf berdasarkan nama kabupaten/kota
gdf = gdf.merge(data_avg[['Kab/Kota', 'Cluster']], how='left', left_on='kabkot', right_on='Kab/Kota')

# Buat peta interaktif menggunakan Folium
m = folium.Map(location=[-7.54, 112.23], zoom_start=8)

# Warna untuk kluster
cluster_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow'}
cluster_descriptions = {
    0: 'Produksi Sangat Rendah',
    1: 'Produksi Sedang',
    2: 'Produksi Tinggi',
    3: 'Produksi Sangat Tinggi'
}

# Tambahkan wilayah berdasarkan geometri dan kluster
for _, row in gdf.iterrows():
    folium.GeoJson(
        row['geometry'],
        style_function=lambda x, color=cluster_colors[row['Cluster']]: {
            'fillColor': color,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6,
        },
        tooltip=folium.Tooltip(f"{row['kabkot']} (Cluster {row['Cluster']})"),
    ).add_to(m)

# Tambahkan legenda
legend_html = """
<div style="position: fixed; 
            bottom: 50px; left: 50px; width: 250px; height: 180px; 
            background-color: white; z-index:9999; font-size:14px;
            border:2px solid grey; padding: 10px;">
<strong>Cluster Legend</strong><br>
<span style="color:red;">●</span> Cluster 0 - Produksi Sangat Rendah<br>
<span style="color:blue;">●</span> Cluster 1 - Produksi Sedang<br>
<span style="color:green;">●</span> Cluster 2 - Produksi Tinggi<br>
<span style="color:yellow;">●</span> Cluster 3 - Produksi Sangat Tinggi<br>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Tampilkan peta dalam Streamlit
st_folium(m, width=700)

