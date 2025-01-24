import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")



print(customers.head())
print(products.info())
print(transactions.describe())


merged = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")





customer_features = merged.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'Region': 'first'
}).reset_index()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
customer_features['Region'] = le.fit_transform(customer_features['Region'])


from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(customer_features.iloc[:, 1:])
similarity_df = pd.DataFrame(similarity_matrix, index=customer_features['CustomerID'], columns=customer_features['CustomerID'])


lookalike_dict = {}
for idx, row in enumerate(similarity_matrix):
    similar_indices = row.argsort()[-4:-1][::-1]
    lookalike_dict[customer_features.iloc[idx]['CustomerID']] = [
        (customer_features.iloc[i]['CustomerID'], row[i]) for i in similar_indices
    ]


import csv
with open("FirstName_LastName_Lookalike.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['CustomerID', 'SimilarCustomers'])
    for key, value in lookalike_dict.items():
        writer.writerow([key, value])

clustering_data = customer_features[['TotalValue', 'Quantity', 'Region']]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
customer_features['Cluster'] = kmeans.fit_predict(clustering_data_scaled)

from sklearn.metrics import davies_bouldin_score
db_index = davies_bouldin_score(clustering_data_scaled, customer_features['Cluster'])
print("DB Index:", db_index)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(clustering_data_scaled)
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=customer_features['Cluster'])
plt.title("Customer Clusters")
plt.show()



