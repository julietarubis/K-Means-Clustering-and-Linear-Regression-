import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Create a dataset
data = {
    'Height': [80, 100, 120, 135, 160, 175, 140, 155, 180, 110],
    'Weight': [12, 22, 35, 50, 60, 70, 45, 65, 85, 30]
}

df = pd.DataFrame(data)

# Step 2: Plot the initial data (optional)
plt.scatter(df['Height'], df['Weight'], color='blue')
plt.title("People's Height vs Weight")
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()

# Step 3: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3)  # Choose the number of clusters
df['Cluster'] = kmeans.fit_predict(df[['Height', 'Weight']])

# Step 4: Visualize the clustering
plt.scatter(df['Height'], df['Weight'], c=df['Cluster'], cmap='viridis')
plt.title('K-Means Clustering of People by Height and Weight')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.colorbar(label='Cluster')
plt.show()

# Step 5: Print cluster centers and assigned labels
print("Cluster Centers: \n", kmeans.cluster_centers_)
print("Assigned Clusters: \n", df)
