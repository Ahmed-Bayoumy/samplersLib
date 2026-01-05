import numpy as np
import pytest

# ----------------------------------------------------------------------
# Simple mock predictor – behaves deterministically and returns a fixed
# uncertainty.  It implements the same public API that AdaptiveEnsemble
# expects (fit, predict, uncertainty).
# ----------------------------------------------------------------------
class MockPredictor:
    def __init__(self, bias=0.0, scale=1.0, uncert=0.1):
        self.bias = bias
        self.scale = scale
        self.uncert = uncert
        self.fitted = False

    def fit(self, X, y):
        # just remember that fit was called – no real training
        self.fitted = True

    def predict(self, x):
        # linear function: scale * x + bias
        return self.scale * np.asarray(x) + self.bias

    def uncertainty(self, x):
        # constant uncertainty (same shape as prediction)
        return np.full_like(np.asarray(x), self.uncert, dtype=float)


# ----------------------------------------------------------------------
# Helper to build a list of three diverse mock models
# ----------------------------------------------------------------------
@pytest.fixture
def three_models():
    return [
        MockPredictor(bias=0.0, scale=1.0, uncert=0.1),   # model A
        MockPredictor(bias=1.0, scale=0.5, uncert=0.2),   # model B
        MockPredictor(bias=-0.5, scale=2.0, uncert=0.05)  # model C
    ]


# ----------------------------------------------------------------------
# 1️⃣  Instantiation & basic attributes
# ----------------------------------------------------------------------
def test_instantiation(three_models):
    from samplersLib.predictors import AdaptiveEnsemble

    ae = AdaptiveEnsemble(models=three_models, bandwidth=0.2)

    assert isinstance(ae, AdaptiveEnsemble)
    assert ae.n_models == 3
    # uniform start → each weight = 1/3
    np.testing.assert_allclose(ae.weights, np.full(3, 1/3))


# ----------------------------------------------------------------------
# 2️⃣  fit() forwards to each sub‑model
# ----------------------------------------------------------------------
def test_fit_calls_submodels(three_models):
    from samplersLib.predictors import AdaptiveEnsemble

    ae = AdaptiveEnsemble(models=three_models)
    X = np.arange(5).reshape(-1, 1)
    y = np.arange(5)

    ae.fit(X, y)

    for m in three_models:
        assert m.fitted is True


# ----------------------------------------------------------------------
# 3️⃣  predict() returns weighted average of sub‑model predictions
# ----------------------------------------------------------------------
def test_predict_weighted_average(three_models):
    from samplersLib.predictors import AdaptiveEnsemble

    ae = AdaptiveEnsemble(models=three_models)
    x = np.array([2.0])

    # manual weighted average with uniform weights (1/3 each)
    expected = (
        (1/3) * three_models[0].predict(x) +
        (1/3) * three_models[1].predict(x) +
        (1/3) * three_models[2].predict(x)
    )
    np.testing.assert_allclose(ae.predict(x), expected)


# ----------------------------------------------------------------------
# 4️⃣  uncertainty() combines weighted uncertainties + variance of predictions
# ----------------------------------------------------------------------
def test_uncertainty_combination(three_models):
    from samplersLib.predictors import AdaptiveEnsemble

    ae = AdaptiveEnsemble(models=three_models)
    x = np.array([1.0])

    preds = np.array([m.predict(x) for m in three_models])
    uncerts = np.array([m.uncertainty(x) for m in three_models])

    mean_pred = np.dot(ae.weights, preds)
    variance = np.dot(ae.weights, (preds - mean_pred) ** 2)
    weighted_uncert = np.dot(ae.weights, uncerts)

    expected = np.sqrt(weighted_uncert ** 2 + variance)
    np.testing.assert_allclose(ae.uncertainty(x), expected)


# ----------------------------------------------------------------------
# 5️⃣  predict_with_zscore() – returns list of z‑scores when observations supplied
# ----------------------------------------------------------------------
def test_predict_with_zscore(three_models):
    from samplersLib.predictors import AdaptiveEnsemble

    ae = AdaptiveEnsemble(models=three_models)

    # two test points
    xps = [np.array([0.0]), np.array([2.0])]
    # true observations (chosen arbitrarily)
    y_obs = [0.5, 3.0]

    # compute expected z‑scores manually
    z_expected = []
    for x, y in zip(xps, y_obs):
        preds = np.array([m.predict(x) for m in three_models])
        uncerts = np.array([m.uncertainty(x) for m in three_models])

        mean = np.dot(ae.weights, preds)
        var = np.dot(ae.weights, (preds - mean) ** 2)
        weighted_unc_sq = np.dot(ae.weights, uncerts ** 2)
        sigma = np.sqrt(weighted_unc_sq + var)

        z_expected.append((y - mean) / sigma)

    z_actual = ae.predict_with_zscore(xps, y_obs)
    np.testing.assert_allclose(z_actual, z_expected)


# ----------------------------------------------------------------------
# 6️⃣  calculate_testing_error() – mean squared error on a set
# ----------------------------------------------------------------------
def test_calculate_testing_error(three_models):
    from samplersLib.predictors import AdaptiveEnsemble

    ae = AdaptiveEnsemble(models=three_models)

    xps = [np.array([0.0]), np.array([1.0]), np.array([2.0])]
    yps = [0.0, 1.5, 4.0]   # arbitrary ground‑truth values

    # manual MSE
    mse = 0.0
    for x, y in zip(xps, yps):
        preds = np.array([m.predict(x) for m in three_models])
        mse += (y - np.dot(ae.weights, preds)) ** 2
    mse /= (len(xps) - 1)   # note: the original code divides by i (last index)

    np.testing.assert_allclose(ae.calculate_testing_error(xps, yps), mse)
