import pandas as pd
from sklearn.cluster import KMeans
import warnings
import matplotlib.pyplot as plt

# Ignore warnings
warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv("ab_march24.csv")
print(data)

# Extract features for clustering
features = data[["A", "B"]]

# Create and fit the KMeans model
model = KMeans(n_clusters=3, random_state=3)
res = model.fit_predict(features)

# Add cluster labels to the DataFrame
data["clusters"] = res
print(data)

# Get cluster centroids
ca = model.cluster_centers_[0]
cb = model.cluster_centers_[1]
cc = model.cluster_centers_[2]
print("ca =", ca)
print("cb =", cb)
print("cc =", cc)

# Filter the data for each cluster
da = data[data.clusters == 0]
db = data[data.clusters == 1]
dc = data[data.clusters == 2]
# These lines filter the DataFrame data to create separate DataFrames for each cluster based on their cluster labels

# Plotting the clusters
plt.figure(figsize=(12, 5))
plt.scatter(da["A"], da["B"], color="red", label="cluster a")
plt.scatter(db["A"], db["B"], color="green", label="cluster b")
plt.scatter(dc["A"], dc["B"], color="blue", label="cluster c")

# Plotting the centroids
plt.scatter(ca[0], ca[1], color="black", marker="x", label="ca " + str(ca))
plt.scatter(cb[0], cb[1], color="black", marker="x", label="cb " + str(cb))
plt.scatter(cc[0], cc[1], color="black", marker="x", label="cc " + str(cc))

# Add legend and display the plot
plt.legend()
plt.show()

# Accept user input and predict the cluster
a = float(input("Enter value of a: "))
b = float(input("Enter value of b: "))
d = [[a, b]]
ans = model.predict(d)

# Determine the cluster
if ans == 0:
    print("cluster a")
elif ans == 1:
    print("cluster b")
else:
    print("cluster c")

# Print the cluster centroids
print(model.cluster_centers_)
ca = model.cluster_centers_[0]
cb = model.cluster_centers_[1]
cc = model.cluster_centers_[2]
print("ca =", ca)
print("cb =", cb)
print("cc =", cc)
