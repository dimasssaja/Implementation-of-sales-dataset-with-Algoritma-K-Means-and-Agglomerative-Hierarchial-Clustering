# Laporan05-Pembelajaran-Mesin

Penjelasan Codingan

Menggunakan Dataset Sales.csv
Algoritma K-Means dan Agglomerative Hierarchial Clustering

#Meng-import Library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Membaca Dataset
data = pd.read_csv("sales.csv")
data

#Memilih data dari dataset untuk digunakan sebagai atribut X
data = data[["W1", "W2"]] #balance x1; purchases= 2x
data.head(10)

#Melihat Ringkasan Statistik
data.describe()

#Menampilkan boxplot dari data W1 DAN W2
fig, ax = plt.subplots()
ax.boxplot(data,
          vert = False, 
          showmeans = True,
          meanline = True, 
          labels = ("W1", "W2"),
          patch_artist = True, 
          medianprops = {"linewidth" : 2, "color" : "red"},
          meanprops = {"linewidth" : 2, "color" : "blue"})
plt.show()

#Menampilkan boxplot dari data W1 DAN W2 setelah outlier dibuang
#q1, q3, dan IQR
kolom = ["W1", "W2"]

Q1 = data[kolom].quantile(0.25)
Q3 = data[kolom].quantile(0.75)
IQR = Q3-Q1
data = data[~((data[kolom]<(Q1 - 1.5 * IQR)) |
             (data[kolom]>(Q3 + 1.5 * IQR))).any(axis = 1)]

plt.boxplot(data[kolom])
plt.xticks([1,2], kolom)
plt.title("Outlier Setelah Dibuang")
plt.show()

#mendeskripsikan data
data.describe()

#melihat informasi dari data
data.info()

#Membuat nilai X
x_array = np.array (data)

#Visualisasi Persebaran Data
plt.scatter(data.W1, data.W2)
plt.show()

# Melakukan Standarisasi data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_array)
plt.scatter(x_scaled[:,0], x_scaled[:,1], s=10)
plt.show()

#Cek Sum Of Square Error Dari Tiap Pembagian Jumlah Cluster
from sklearn.cluster import KMeans
sse = []
index = range(1,10)
for i in index :
    kmeans = KMeans(n_clusters=i, random_state=0, n_init=10)  # Ganti 10 dengan angka yang sesuai
    kmeans.fit(x_scaled)
    sse_ = kmeans.inertia_
    sse.append(sse_)
    print (i,sse_)

#Menampilkan data dari Square Means error
plt.plot(index, sse)
plt.show()

#Membuat Model
kmeans = KMeans (n_clusters = 3, random_state = 0, n_init=10)
kmeans.fit(x_scaled)

#Melihat Cluster Pusat
kmeans.cluster_centers_

#Visualisasi Perseberan Data Setelah Clustering
output = plt.scatter(x_scaled[:,0],x_scaled[:,1], s=10, c = kmeans.labels_) #Datanya

centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c="red", s=10) #centroid
plt.title("KMeans Clustering sales")
plt.xlabel("W1")
plt.xlabel("W2")
plt.colorbar(output)
plt.show()

#Evaluasi Model
from sklearn.metrics import davies_bouldin_score
labels = kmeans.labels_
davies_bouldin_score(x_scaled, labels)

#Agglomerative Hierarchical Clustering dari data w1 dan w2
plt.scatter(data['W1'], data['W2'], s=50)
plt.show()

# memangil data w1 dan w2
data = np.asarray(data[['W1', 'W2']])
print(data)

#Import library untuk hierarchial clustering
import scipy.cluster.hierarchy as sch

#menampilkan dendogram secara ward
plt.title('Dendrogram')
dendogram = sch.dendrogram(sch.linkage(data, method='ward'))

#menampilkan dendogram secara average
plt.title('Dendogram')
dendrogram = sch.dendrogram(sch.linkage(data, method='average'))

# Visualisasi persebaran data euclidean
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=5)
output_euclidean = ac.fit_predict(data)

plt.scatter(data[:,0], data[:,1], c=output_euclidean, s=50, cmap='rainbow')
plt.show()

# Visualisasi data Manhattan
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=5, affinity='manhattan', linkage='complete')
output_manhattan = ac.fit_predict(data)

plt.scatter(data[:,0], data[:,1], c=output_manhattan, s=20, cmap='rainbow')
plt.show()

#Pengecekan Akurasi Hierarchial Clustering
from sklearn.metrics import davies_bouldin_score
print(davies_bouldin_score(data, output_euclidean))
print(davies_bouldin_score(data, output_manhattan))
