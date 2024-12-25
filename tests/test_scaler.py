import numpy as np
from graddft_qnn.standard_scaler import StandardScaler
from numpy.testing import assert_almost_equal, assert_array_almost_equal

# Make some data to be used many times
rng = np.random.RandomState(0)
n_features = 30
n_samples = 1000
offsets = rng.uniform(-1, 1, size=n_features)
scales = rng.uniform(1, 10, size=n_features)
X_2d = rng.randn(n_samples, n_features) * scales + offsets
X_1row = X_2d[0, :].reshape(1, n_features)
X_1col = X_2d[:, 0].reshape(n_samples, 1)


def _check_dim_1axis(a):
    return np.asarray(a).shape[0]


def test_standard_scaler_1d():
    # Test scaling of dataset along single axis
    for X in [X_1row, X_1col]:
        scaler = StandardScaler()
        X_scaled = scaler.fit(X).transform(X)

        if isinstance(X, list):
            X = np.array(X)  # cast only after scaling done

        if _check_dim_1axis(X) == 1:
            assert_almost_equal(scaler.mean_, X.ravel(), decimal=6)
            assert_array_almost_equal(X_scaled.mean(axis=0), np.zeros_like(n_features), decimal=6)
            assert_array_almost_equal(X_scaled.std(axis=0), np.zeros_like(n_features), decimal=6)
        else:
            assert_almost_equal(scaler.mean_, X.mean(), decimal=6)
            assert_array_almost_equal(X_scaled.mean(axis=0), np.zeros_like(n_features), decimal=6)
            assert_array_almost_equal(X_scaled.mean(axis=0), 0.0, decimal=6)
            assert_array_almost_equal(X_scaled.std(axis=0), 1.0, decimal=6)

        # check inverse transform
        X_scaled_back = scaler.inverse_transform(X_scaled)
        assert_array_almost_equal(X_scaled_back, X)

    # Constant feature
    X = np.ones((5, 1))
    scaler = StandardScaler()
    X_scaled = scaler.fit(X).transform(X)
    assert_almost_equal(scaler.mean_, 1.0)
    assert_array_almost_equal(X_scaled.mean(axis=0), 0.0)
    assert_array_almost_equal(X_scaled.std(axis=0), 0.0)