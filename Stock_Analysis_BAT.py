import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import euclidean
from sklearn.cluster import AgglomerativeClustering
from Practices.hierarchical_clustering import dendogram

bat_stock_data = pd.read_csv('British_American_Tobacco_Stock Data.csv')
bat_data = bat_stock_data.iloc[:, 4:6].values
print(bat_data)

plt.figure(figsize=(10, 7))
plt.title('British American Tobacco Market Regimes (Close Price vs. Volume)')
dendogram = sch.dendrogram(sch.linkage(bat_data, method='ward'))
plt.show()

cluster_model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
cluster_model.fit_predict(bat_data)

plt.figure(figsize=(10, 7))
plt.scatter(bat_data[:, 0], bat_data[:, 1], c=cluster_model.labels_, cmap="viridis", alpha=0.6)
plt.title("British American Tobacco Market Regimes (Close Price vs. Volume)")

plt.xlabel('Close Price')
plt.ylabel('Volume')
plt.grid(True)
plt.show()