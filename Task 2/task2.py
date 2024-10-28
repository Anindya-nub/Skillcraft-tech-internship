import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('Task2/mall_customers.csv')

# Handle categorical variables (convert 'Gender' to numerical)
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Extract features (Age, Annual Income, and Spending Score)
features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Apply KMeans clustering (choose k=5 based on the Elbow Method)
k = 5  # Adjust the number of clusters based on the Elbow plot
kmeans = KMeans(n_clusters=k, random_state=0)
data['cluster'] = kmeans.fit_predict(scaled_features)

# Print the first few rows with cluster assignments
print(data.head())

# Visualize clusters for all features
plt.figure(figsize=(10, 6))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['cluster'], cmap='viridis', s=50)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Clusters by Annual Income and Spending Score')
plt.colorbar(label='Cluster')
plt.show()

# Save the cluster results to a CSV file
submission = data[['CustomerID', 'cluster']]  # Replace 'CustomerID' with the correct column name for unique identifiers
submission.rename(columns={'CustomerID': 'Customer_ID', 'cluster': 'Cluster_Label'}, inplace=True)
submission.to_csv('Task2/customer_clusters.csv', index=False)
