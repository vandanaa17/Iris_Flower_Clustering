#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score


# In[5]:


iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df["target"] = iris.target
df.head()


# In[6]:


X = iris.data
kmeans = KMeans(n_clusters=3,random_state=42,n_init=10)
clusters = kmeans.fit_predict(X)
df["cluster"]=clusters
df.head(10)


# In[7]:


# Visualize the clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=clusters,
    cmap="viridis",
    s=50
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering of Iris Dataset")

plt.show()


# In[8]:


# Compare predicted clusters with true labels

comparison = pd.crosstab(
    df["target"],
    df["cluster"],
    rownames=["True Label"],
    colnames=["Predicted Cluster"]
)

print(comparison)


# In[9]:


# Convert true labels to species names
df["species"] = df["target"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

# Compare species with predicted clusters
comparison = pd.crosstab(
    df["species"],
    df["cluster"],
    rownames=["True Species"],
    colnames=["Predicted Cluster"]
)

print(comparison)


# In[10]:


# Apply PCA to reduce 4 features to 2 dimensions
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

# Display the first 5 transformed rows
print(X_pca[:5])


# In[11]:


# Visualize K-Means clusters using PCA
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis",
    s=50
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Iris Clusters using PCA")

plt.show()


# In[12]:


from sklearn.metrics import silhouette_score


# In[13]:


# Calculate silhouette score
score = silhouette_score(X, clusters)

print("Silhouette Score:", score)


# In[14]:


print("Number of clusters:", 3)
print("Silhouette Score:", score)
print("\nClustering completed successfully!")
print("K-Means grouped the Iris flowers into 3 clusters.")
print("PCA was used to visualize the clusters in 2 dimensions.")


# In[ ]:




