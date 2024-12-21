import grad_dft as gd
from jaxlib.xla_extension import ArrayImpl
from jaxtyping import Scalar
from flax import linen as nn
# from sklearn.preprocessing import StandardScale

class QNNFunctional(gd.Functional):
    @nn.compact
    def __call__(self, coefficient_inputs) -> Scalar:
        r"""Where the functional is called, mapping the density to the energy.
        Expected to be overwritten by the inheriting class.
        Should use the _integrate() helper function to perform the integration.

        Parameters
        ---------
        inputs: inputs to the function f

        Returns
        -------
        Union[Array, Scalar]
        """
        # why self an argument here?
        return self.coefficients(coefficient_inputs)

    def dim_reduction(self, original_array: ArrayImpl):
        # Standardize the data
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(X)

        # Compute SVD
        U, S, Vt = svd(X_standardized, full_matrices=False)

        # Project the data onto the first two principal components
        X_pca_svd = np.dot(X_standardized, Vt.T[:, :2])

        # Plot the PCA results
        plt.figure(figsize=(8, 6))
        for target in np.unique(y):
            plt.scatter(X_pca_svd[y == target, 0], X_pca_svd[y == target, 1], label=iris.target_names[target])

        plt.title('PCA using SVD of Iris Dataset')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        plt.show()
