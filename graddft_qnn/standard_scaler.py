
import jax.numpy as np


class StandardScaler:
    """
    A version of standard scaler implemented in jnp
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
        Calculate the mean and standard deviation for each feature.
        Parameters:
            X (ndarray): Input data of shape (n_samples, n_features).
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        """
        Standardize features by removing the mean and scaling to unit variance.
        Parameters:
            X (ndarray): Input data of shape (n_samples, n_features).
        Returns:
            X_scaled (ndarray): Scaled data of the same shape as X.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        X -= self.mean_
        if np.isclose(self.std_, np.zeros_like(self.std_)).any():
            return X
        return X / self.std_

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        Parameters:
            X (ndarray): Input data of shape (n_samples, n_features).
        Returns:
            X_scaled (ndarray): Scaled data of the same shape as X.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        if np.isclose(self.std_, np.zeros_like(self.std_)).any():
            return X + self.mean_
        return X * self.std_ + self.mean_
