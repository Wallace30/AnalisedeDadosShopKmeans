import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('Mall_Customers.csv')

data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

X = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

inertia = []
silhouette_scores = []
K = range(2, 11)  
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-', markersize=8)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo para Determinação do Número Ideal de Clusters')
plt.xticks(K)
plt.grid(True)
plt.show()

ideal_clusters = K[silhouette_scores.index(max(silhouette_scores))]

num_clusters = int(input(f"Quantos clusters você deseja criar? (Sugerido: {ideal_clusters}) "))

kmeans = KMeans(n_clusters=num_clusters, random_state=0)
data['Cluster'] = kmeans.fit_predict(X)

centroids = kmeans.cluster_centers_

cluster_summary = data.groupby('Cluster').mean(numeric_only=True)[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
cluster_summary['Size'] = data['Cluster'].value_counts().sort_index()
print(cluster_summary)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

colors = plt.get_cmap('tab20', num_clusters)

for cluster in range(num_clusters):
    ax.scatter(
        data.loc[data['Cluster'] == cluster, 'Age'],
        data.loc[data['Cluster'] == cluster, 'Annual Income (k$)'],
        data.loc[data['Cluster'] == cluster, 'Spending Score (1-100)'],
        s=50, c=[colors(cluster)], label=f'Cluster {cluster}'
    )

ax.scatter(centroids[:, 1], centroids[:, 2], centroids[:, 3], s=200, c='yellow', marker='X', edgecolor='black', label='Centróides')


ax.set_xlabel('Idade')
ax.set_ylabel('Renda Anual (k$)')
ax.set_zlabel('Pontuação de Gastos (1-100)')
ax.set_title('Clusters de Clientes')
ax.legend()
plt.show()
