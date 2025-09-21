# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 2: Load Data
df = pd.read_csv("E:\\ml pbl\\Mall_Customers.csv")

  # Ye file folder me honi chahiye

# Step 3: Preprocessing
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
features = df[['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 4: Elbow Method (Best Cluster Find)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# Step 5: Apply KMeans
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
df['Cluster'] = clusters

# Step 6: Visualize Clusters
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=clusters, cmap='rainbow', s=50)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Customer Segments')
plt.show()

# Step 7: Cluster Summary
print("\nCluster-wise Summary:")
print(df.groupby('Cluster').mean())

# Step 8: Save Result
df.to_csv('customer_segments.csv', index=False)
print("Segmented data saved to 'customer_segments.csv'")
