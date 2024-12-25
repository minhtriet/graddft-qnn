import jax.numpy as np


def pca(X, n_components=2):
    """
    Perform PCA on the dataset X and reduce its dimensionality to n_components.

    Parameters:
    - X: Normalized input data matrix of shape (n_samples, n_features)
    - n_components: Number of principal components to retain (default is 2)

    Returns:
    - X_pca: Transformed data in the reduced space
    - components: The principal components (eigenvectors)
    - explained_variance: The variance explained by each of the principal components
    """

    cov_matrix = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort in descending order
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    selected_components = eigenvectors[:, :n_components]

    X_pca = X.dot(selected_components)

    # Explained variance (proportion of variance explained by each component)
    explained_variance = eigenvalues[:n_components] / np.sum(eigenvalues)

    return X_pca, selected_components, explained_variance
