import numpy as np
import pytest

from exhbma import ExhaustiveLinearRegression, StandardScaler, inverse
from exhbma.sampling import (
    IndicatorSampling,
    InversePrior,
    SamplingAttributes,
    SamplingVariables,
)


@pytest.fixture()
def empty_sampler():
    sampler = IndicatorSampling(X=np.zeros((5, 2)), y=np.zeros(5))
    return sampler


def check_sampling_entity_by_dict(actual, expect):
    """
    Utility function to check SamplingVariables and SamplingAttributes.
    """
    for key, val in expect.dict().items():
        if isinstance(val, float):
            assert getattr(actual, key) == pytest.approx(val)
        else:
            assert np.allclose(getattr(actual, key), val)


def test_extract_X(empty_sampler):
    X = np.array([[1, 2, 3], [2, 2, 2], [3, 4, 5], [4, 4, 4]])
    indicator = [1, 0, 1]

    expect = np.array([[1, 3], [2, 2], [3, 5], [4, 4]])

    actual = empty_sampler._extract_X(X=X, indicator=indicator)
    assert np.all(actual == expect)


def test_calculate_log_model_prior(empty_sampler):
    alpha = 0.3
    indicator = [1, 0, 1, 1, 0]

    expect = 3 * np.log(0.3) + 2 * np.log(0.7)

    actual = empty_sampler._calculate_log_model_prior(indicator=indicator, alpha=alpha)

    assert actual == expect


def test_calculate_inv_det(empty_sampler):
    sigma_noise = 1.0
    sigma_coef = 0.5
    X = np.array([[2, -(3 ** 0.5) * 2], [3 ** 0.5, 1]])

    expect_log_det = np.log(10)
    A = sigma_coef ** 2 * np.dot(X, X.T) + sigma_noise ** 2 * np.eye(2)

    actual_log_det, actual_inv = empty_sampler._calculate_inv_det(
        X=X, sigma_noise=sigma_noise, sigma_coef=sigma_coef
    )

    assert actual_log_det == pytest.approx(expect_log_det)
    assert np.allclose(np.dot(A, actual_inv), np.eye(2))


def test_calculate_inv_det_for_null_matrix(empty_sampler):
    sigma_noise = 2.0
    sigma_coef = 0.5
    X: np.ndarray = np.array([[], [], []])
    assert X.shape == (3, 0)

    expect_log_det = np.log(64)
    expect_inv = np.eye(3) / 4

    actual_log_det, actual_inv = empty_sampler._calculate_inv_det(
        X=X, sigma_noise=sigma_noise, sigma_coef=sigma_coef
    )

    assert actual_log_det == pytest.approx(expect_log_det)
    assert np.allclose(actual_inv, expect_inv)


def test_calculate_exact_sampling_attributes(empty_sampler):
    sigma_noise = 1.0
    sigma_coef = 0.5
    X = np.array([[1, 2, 3, 4], [2, 2, 2, 2], [3, 4, 5, 6], [4, 4, 4, 4]])
    indicator = [1, 1, 1, 0]
    extract_X = np.array([[1, 2, 3], [2, 2, 2], [3, 4, 5], [4, 4, 4]])

    sv = SamplingVariables(
        indicator=indicator, sigma_noise=sigma_noise, sigma_coef=sigma_coef
    )

    log_det, inv = empty_sampler._calculate_inv_det(
        X=extract_X, sigma_noise=sigma_noise, sigma_coef=sigma_coef
    )
    expect = SamplingAttributes(
        log_model_prior=empty_sampler._calculate_log_model_prior(
            indicator=indicator, alpha=empty_sampler.alpha
        ),
        log_det=log_det,
        inv=inv,
    )

    actual = empty_sampler._calculate_exact_sampling_attributes(X=X, sv=sv)

    check_sampling_entity_by_dict(actual=actual, expect=expect)


def test_calculate_log_prob(empty_sampler):
    """
    Since `_calculate_log_model_prior` is tested,
    use it to calculate expected value.
    """
    sigma_noise = 1.0
    sigma_coef = 0.5
    X = np.array([[1, 2, 3, 4], [2, 2, 2, 2], [3, 4, 5, 6], [4, 4, 4, 4]])
    y = np.array([-3, 2, 4, -3])
    indicator = [1, 1, 1, 0]
    extract_X = np.array([[1, 2, 3], [2, 2, 2], [3, 4, 5], [4, 4, 4]])

    A = sigma_coef ** 2 * np.dot(extract_X, extract_X.T) + sigma_noise ** 2 * np.eye(4)

    log_prior = empty_sampler._calculate_log_model_prior(
        indicator=indicator, alpha=empty_sampler.alpha
    )
    log_likelihood = (
        -len(y) / 2 * np.log(2 * np.pi)
        - np.log(np.linalg.det(A)) / 2
        - np.dot(y, np.dot(np.linalg.inv(A), y)) / 2
    )
    expect = log_likelihood + log_prior

    sv = SamplingVariables(
        indicator=indicator, sigma_noise=sigma_noise, sigma_coef=sigma_coef
    )
    sa = empty_sampler._calculate_exact_sampling_attributes(X=X, sv=sv)
    actual = empty_sampler._calculate_log_prob(y=y, sa=sa)
    assert actual == pytest.approx(expect)


def test_update_one_index_0to1(empty_sampler):
    sigma_noise = 1.0
    sigma_coef = 0.5
    X = np.array([[1, 2, 3, 4], [2, 2, 2, 2], [3, 4, 5, 6], [4, 4, 4, 4]])
    y = np.array([-3, 2, 4, -3])
    indicator = [1, 0, 1, 0]
    index = 1

    sv = SamplingVariables(
        indicator=indicator, sigma_noise=sigma_noise, sigma_coef=sigma_coef
    )
    sa = empty_sampler._calculate_exact_sampling_attributes(X=X, sv=sv)

    new_indicator = [1, 1, 1, 0]
    expect_sv = SamplingVariables(
        indicator=new_indicator, sigma_noise=sigma_noise, sigma_coef=sigma_coef
    )
    expect_sa = empty_sampler._calculate_exact_sampling_attributes(X=X, sv=expect_sv)

    actual_sa = empty_sampler._update_one_index(index=index, y=y, X=X, sv=sv, sa=sa)

    check_sampling_entity_by_dict(actual=actual_sa, expect=expect_sa)


def test_update_one_index_1to0(empty_sampler):
    sigma_noise = 1.0
    sigma_coef = 0.5
    X = np.array([[1, 2, 3, 4], [2, 2, 2, 2], [3, 4, 5, 6], [4, 4, 4, 4]])
    y = np.array([-3, 2, 4, -3])
    indicator = [1, 1, 1, 0]
    index = 1

    sv = SamplingVariables(
        indicator=indicator, sigma_noise=sigma_noise, sigma_coef=sigma_coef
    )
    sa = empty_sampler._calculate_exact_sampling_attributes(X=X, sv=sv)

    new_indicator = [1, 0, 1, 0]
    expect_sv = SamplingVariables(
        indicator=new_indicator, sigma_noise=sigma_noise, sigma_coef=sigma_coef
    )
    expect_sa = empty_sampler._calculate_exact_sampling_attributes(X=X, sv=expect_sv)

    actual_sa = empty_sampler._update_one_index(index=index, y=y, X=X, sv=sv, sa=sa)

    check_sampling_entity_by_dict(actual=actual_sa, expect=expect_sa)


@pytest.mark.full
def test_indicator_sampling():
    n_data, n_features = 50, 30
    sigma_noise = 0.1

    nonzero_w = [1, -1, 1, -1]
    w = nonzero_w + [0] * (n_features - len(nonzero_w))

    np.random.seed(0)
    X = np.random.randn(n_data, n_features)
    y = np.dot(X, w) + sigma_noise * np.random.randn(n_data)

    x_scaler = StandardScaler(n_dim=2)
    y_scaler = StandardScaler(n_dim=1, scaling=False)
    x_scaler.fit(X)
    y_scaler.fit(y)
    X = x_scaler.transform(X)
    y = y_scaler.transform(y)

    # Fitting with sampling model
    sampler = IndicatorSampling(
        X=X,
        y=y,
        sigma_noise_prior=InversePrior(low=10 ** -3, high=10 ** 1),
        sigma_coef_prior=InversePrior(low=10 ** -3, high=10 ** 1),
    )
    sampler.sample(
        n_burn_in=10 ** 4,
        n_sampling=10 ** 4,
        random_state=0,
        exact_calc_interval=100,
    )

    # Probabilities of Non-zero features should be close to 1
    for i in range(len(nonzero_w)):
        assert sampler.feature_posteriors_[i] > 0.9

    # Probabilities of zero features should be close to 0
    for i in range(len(nonzero_w), n_features):
        assert sampler.feature_posteriors_[i] < 0.1


@pytest.mark.full
def test_indicator_sampling_compared_to_exhaustive_search():
    """
    Indicator posteriors caluculated by ExhaustiveLinearRegression
    and IndicatorSampling should be approximately equal.
    """
    n_data, n_features = 40, 8
    sigma_noise = 0.5

    nonzero_w = [1, 0.5, 0.25]
    w = nonzero_w + [0] * (n_features - len(nonzero_w))

    np.random.seed(0)
    X = np.random.randn(n_data, n_features)
    y = np.dot(X, w) + sigma_noise * np.random.randn(n_data)

    x_scaler = StandardScaler(n_dim=2)
    y_scaler = StandardScaler(n_dim=1, scaling=False)
    x_scaler.fit(X)
    y_scaler.fit(y)
    X = x_scaler.transform(X)
    y = y_scaler.transform(y)

    n_sigma_noise_points = 31
    sigma_noise_log_range = [-2, 1]
    n_sigma_coef_points = 41
    sigma_coef_log_range = [-2, 2]

    # Fitting with ExhaustiveLinearRegression
    reg = ExhaustiveLinearRegression(
        sigma_noise_points=inverse(
            np.logspace(
                sigma_noise_log_range[0], sigma_noise_log_range[1], n_sigma_noise_points
            ),
        ),
        sigma_coef_points=inverse(
            np.logspace(
                sigma_coef_log_range[0], sigma_coef_log_range[1], n_sigma_coef_points
            ),
        ),
    )
    reg.fit(X, y)

    # Fitting with IndicatorSampling
    sampler = IndicatorSampling(
        X=X,
        y=y,
        sigma_noise_prior=InversePrior(
            low=10 ** sigma_noise_log_range[0], high=10 ** sigma_noise_log_range[1]
        ),
        sigma_coef_prior=InversePrior(
            low=10 ** sigma_coef_log_range[0], high=10 ** sigma_coef_log_range[1]
        ),
    )
    sampler.sample(
        n_burn_in=10 ** 4,
        n_sampling=10 ** 4,
        random_state=0,
        exact_calc_interval=100,
    )

    assert sampler.feature_posteriors_ == pytest.approx(
        reg.feature_posteriors_, rel=1e-1
    )
