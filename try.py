import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import datasets
# Load the dataset
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add target and class to DataFrame
df['target'] = iris.target
df['class'] = df.target.apply(lambda x: iris.target_names[x])
# Scaling
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.drop(['target', 'class'], axis=1)), columns=df.columns[:-2])
df_scaled['target'] = df['target']
df_scaled['class'] = df['class']
X = df_scaled.drop(['target', 'class'], axis=1).values
covariance_matrix = np.cov(X.T)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
# Sort the eigenvalues and corresponding eigenvectors
eigenvalue_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[eigenvalue_indices]
sorted_eigenvectors = eigenvectors[:, eigenvalue_indices]
eigenvectors_subset = sorted_eigenvectors[:, :2]
X_pca = X.dot(eigenvectors_subset)
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_scaled['target'])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()
# Calculate the explained variance
explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)

# Calculate the cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance)

# Plot the cumulative explained variance
plt.figure(figsize=(8,6))
plt.plot(range(len(cumulative_explained_variance)), cumulative_explained_variance)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs Number of Principal Components')
plt.show()

# Select the appropriate number of principal components
n_components = np.argmax(cumulative_explained_variance > 0.95) + 1
print(f"The appropriate number of principal components is {n_components}")