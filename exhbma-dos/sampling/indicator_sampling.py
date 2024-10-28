from typing import Any, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from scipy.special import loggamma
from tqdm import tqdm

from .prior import BasePrior, InversePrior


class SamplingVariables(BaseModel):
    indicator: List[int] = Field(..., description="Indicator vector of the model.")
    sigma_noise: float = Field(
        ..., description="Standard deviation of observation noise."
    )
    sigma_coef: float = Field(
        ..., description="Standard deviation of prior for coefficient."
    )


class SamplingAttributes(BaseModel):
    log_model_prior: float = Field(..., description="Log value of prior for the model.")
    log_det: float = Field(
        ..., description="Log value of determinant of variance-covariance matrix."
    )
    inv: Any = Field(..., description="Inverse matrix of variance-covariance matrix.")


class IndicatorSampling(object):
    """
    Sampling calculation for linear regression model.

    Sampling variables
    - indicator vector
    - sigma_noise
    - sigma_coef

    Parameters
    ----------
    X : np.ndarray with shape (n_data, n_features)
        Feature matrix. Each row corresponds to single data.

    y : np.ndarray with shape  (n_data,)
        Target value vector.

    alpha: float (default: 0.5)
        Alpha parameter is fixed to this value.

    sigma_noise_prior: func
        Function of prior distribution for sigma_noise

    sigma_coef_prior: func
        Function of prior distribution for sigma_coef


    Attributes
    ----------
    n_features_in_: int
        Number of features seen during fit.

    feature_posteriors_: List[float]
        Posterior probabilities for each feature.

    """

    def __init__(
        self,
        X,
        y,
        alpha: float = 0.5,
        step_ratio: float = 0.1,
        init_indicator: Optional[List[int]] = None,
        init_sigma_noise=1.0,
        init_sigma_coef=1.0,
        sigma_noise_prior: Optional[BasePrior] = None,
        sigma_coef_prior: Optional[BasePrior] = None,
    ):
        self.X = X
        self.y = y
        self.alpha = alpha

        self.n_features_in_ = X.shape[1]

        self.step_ratio = step_ratio
        self.init_indicator = (
            self._sample_indicator_from_prior()
            if init_indicator is None
            else init_indicator
        )
        self.init_sigma_noise = init_sigma_noise
        self.init_sigma_coef = init_sigma_coef

        self.sigma_noise_prior: BasePrior = (
            InversePrior() if sigma_noise_prior is None else sigma_noise_prior
        )
        self.sigma_coef_prior: BasePrior = (
            InversePrior() if sigma_coef_prior is None else sigma_coef_prior
        )

        self.log_prob_sequence_: List[float] = []
        self.indicator_sequence_: List[List[int]] = []
        self.sigma_noise_sequence_: List[float] = []
        self.sigma_coef_sequence_: List[float] = []

    def _sample_indicator_from_prior(self) -> List[int]:
        indicator = np.random.choice(  # type: ignore
            2, self.n_features_in_, [1 - self.alpha, self.alpha]
        ).tolist()
        return indicator

    def _update_one_index(
        self,
        index: int,
        y: np.ndarray,
        X: np.ndarray,
        sv: SamplingVariables,
        sa: SamplingAttributes,
    ) -> SamplingAttributes:
        # 計算量: O(n^2)
        sign = 1 - sv.indicator[index] * 2

        v = X[:, index] * sv.sigma_coef
        u = np.dot(sa.inv, v)
        vu = np.dot(v, u)

        new_log_model_prior = sa.log_model_prior + sign * (
            np.log(self.alpha) - np.log(1 - self.alpha)
        )
        new_inv = sa.inv + np.outer(u, u / ((-1) * sign - vu))
        new_log_det = sa.log_det + np.log(1 + sign * vu)
        new_sa = SamplingAttributes(
            log_model_prior=new_log_model_prior,
            log_det=new_log_det,
            inv=new_inv,
        )
        return new_sa

    def _binary_sample(self, log_w0: float, log_w1: float):
        """Sample from weight = exp(log_w0) : exp(log_w1)"""
        r = np.random.rand()
        threshold = 1 / (1 + np.exp(log_w1 - log_w0))
        return 1 if r > threshold else 0

    def _sample_indicator(
        self, sv: SamplingVariables, sa: SamplingAttributes
    ) -> Tuple[SamplingVariables, SamplingAttributes]:
        for index in np.random.permutation(self.n_features_in_):
            # Calculate log weight
            current_log_w = sa.log_model_prior + self._calculate_log_prob(
                y=self.y, sa=sa
            )

            new_sa = self._update_one_index(
                index=index, y=self.y, X=self.X, sv=sv, sa=sa
            )
            new_log_w = new_sa.log_model_prior + self._calculate_log_prob(
                y=self.y, sa=new_sa
            )

            val = self._binary_sample(log_w0=current_log_w, log_w1=new_log_w)
            if val == 1:
                sa = new_sa
                sv.indicator[index] = 1 - sv.indicator[index]
        return sv, sa

    def _sample_sigma(self, sigma: float, step_ratio: float = 1.0):
        """
        Sample sigma from gamma distribution
        """
        scale = step_ratio
        shape = sigma / scale
        x = np.random.gamma(shape, scale)
        return x

    def _calculate_sigma_sampler_log_probability(
        self, sigma_from: float, sigma_to: float, step_ratio: float = 1.0
    ):
        scale = step_ratio
        shape = sigma_from / scale
        x = sigma_to
        log_prob = (
            (shape - 1) * np.log(x)
            - x / scale
            - shape * np.log(scale)
            - loggamma(shape)
        )
        return log_prob

    def _sample_sigma_noise(
        self, sv: SamplingVariables, sa: SamplingAttributes
    ) -> Tuple[SamplingVariables, SamplingAttributes]:
        cand_sigma_noise = self._sample_sigma(
            sv.sigma_noise, step_ratio=self.step_ratio
        )

        pre = self._calculate_sigma_sampler_log_probability(
            sigma_from=sv.sigma_noise,
            sigma_to=cand_sigma_noise,
            step_ratio=self.step_ratio,
        )
        pre += self._calculate_log_prob(y=self.y, sa=sa)
        pre += np.log(self.sigma_noise_prior.prob(sv.sigma_noise))

        post = self._calculate_sigma_sampler_log_probability(
            sigma_from=cand_sigma_noise,
            sigma_to=sv.sigma_noise,
            step_ratio=self.step_ratio,
        )
        new_sv = SamplingVariables(
            indicator=sv.indicator,
            sigma_noise=cand_sigma_noise,
            sigma_coef=sv.sigma_coef,
        )
        new_sa = self._calculate_exact_sampling_attributes(X=self.X, sv=new_sv)
        post += self._calculate_log_prob(y=self.y, sa=new_sa)
        post += np.log(self.sigma_noise_prior.prob(new_sv.sigma_noise))

        if np.log(np.random.rand()) < post - pre:
            sv, sa = new_sv, new_sa
        return sv, sa

    def _sample_sigma_coef(
        self, sv: SamplingVariables, sa: SamplingAttributes
    ) -> Tuple[SamplingVariables, SamplingAttributes]:
        cand_sigma_coef = self._sample_sigma(sv.sigma_coef, step_ratio=self.step_ratio)

        pre = self._calculate_sigma_sampler_log_probability(
            sigma_from=sv.sigma_coef,
            sigma_to=cand_sigma_coef,
            step_ratio=self.step_ratio,
        )
        pre += self._calculate_log_prob(y=self.y, sa=sa)
        pre += np.log(self.sigma_coef_prior.prob(sv.sigma_coef))

        post = self._calculate_sigma_sampler_log_probability(
            sigma_from=cand_sigma_coef,
            sigma_to=sv.sigma_coef,
            step_ratio=self.step_ratio,
        )
        new_sv = SamplingVariables(
            indicator=sv.indicator,
            sigma_noise=sv.sigma_noise,
            sigma_coef=cand_sigma_coef,
        )
        new_sa = self._calculate_exact_sampling_attributes(X=self.X, sv=new_sv)
        post += self._calculate_log_prob(y=self.y, sa=new_sa)
        post += np.log(self.sigma_coef_prior.prob(new_sv.sigma_coef))

        if np.log(np.random.rand()) < post - pre:
            sv, sa = new_sv, new_sa
        return sv, sa

    def _calculate_exact_sampling_attributes(
        self, X: np.ndarray, sv: SamplingVariables
    ) -> SamplingAttributes:
        # NOTE: 計算量は O(n^3)
        log_model_prior = self._calculate_log_model_prior(
            indicator=sv.indicator, alpha=self.alpha
        )

        log_det, inv = self._calculate_inv_det(
            X=self._extract_X(X=X, indicator=sv.indicator),
            sigma_noise=sv.sigma_noise,
            sigma_coef=sv.sigma_coef,
        )

        sa = SamplingAttributes(
            log_model_prior=log_model_prior,
            log_det=log_det,
            inv=inv,
        )
        return sa

    def _extract_X(self, X: np.ndarray, indicator: List[int]):
        return X[:, np.array(indicator) == 1]

    def _calculate_log_model_prior(self, indicator: List[int], alpha: float) -> float:
        r"""
        Model prior specified by \alpha:
            p(c) = \prod_{i=1}^p \alpha^{c_i} (1-\alpha)^{1-c_i}
        """
        n_features = len(indicator)
        n_used = sum(indicator)
        log_model_prior = n_used * np.log(alpha) + (n_features - n_used) * np.log(
            1 - alpha
        )
        return log_model_prior

    def _calculate_inv_det(
        self, X: np.ndarray, sigma_noise: float, sigma_coef: float
    ) -> Tuple[float, List[List[float]]]:
        r"""
        Calculate inverse and determinant of matrix A in likelihood function
            A = \sigma_{coef} ** 2 X X^T + \sigma_{noise} ** 2 I

        Use SVD to stabilize the calculation.
        """
        n_data = X.shape[0]

        u, s, vh = np.linalg.svd(X, full_matrices=True)
        eigvals = np.ones(n_data) * sigma_noise ** 2
        eigvals[: len(s)] += (sigma_coef ** 2) * (s ** 2)

        log_det = np.sum(np.log(eigvals))
        inv = np.dot(u, np.dot(np.diag(1 / eigvals), u.T))
        return log_det, inv  # type: ignore

    def _calculate_log_likelihood(self, y, inv, log_det) -> float:
        r"""
        Calculate log likelihood p(y|X, \sigma_{noise}, \sigma_{coef})
        from pre-calculated values.
            p(y|X, \sigma_{noise}, \sigma_{coef}) = N(y|0, A)
            where A = \sigma_{coef} ** 2 X X^T + \sigma_{noise} ** 2 I
        """
        n_data = len(y)
        val = -n_data / 2 * np.log(2 * np.pi)
        val -= log_det / 2
        val -= np.dot(y, np.dot(inv, y)) / 2
        return val

    def _calculate_log_prob(self, y: np.ndarray, sa: SamplingAttributes) -> float:
        r"""
        Calculate log value of joint distribution
        p(y, c|X, \sigma_{noise}, \sigma_{coef}).
        """
        # 計算量: O(n^2)
        log_likelihood = self._calculate_log_likelihood(
            y=y, inv=sa.inv, log_det=sa.log_det
        )
        log_p = log_likelihood + sa.log_model_prior
        return log_p

    def _record_sequence(self, sv: SamplingVariables, sa: SamplingAttributes):
        self.indicator_sequence_.append(sv.indicator)
        self.sigma_noise_sequence_.append(sv.sigma_noise)
        self.sigma_coef_sequence_.append(sv.sigma_coef)
        self.log_prob_sequence_.append(self._calculate_log_prob(y=self.y, sa=sa))

    def _init_sampling(self) -> Tuple[SamplingVariables, SamplingAttributes]:
        sv = SamplingVariables(
            indicator=self.init_indicator,
            sigma_noise=self.init_sigma_noise,
            sigma_coef=self.init_sigma_coef,
        )
        sa = self._calculate_exact_sampling_attributes(X=self.X, sv=sv)
        return sv, sa

    def _sample(
        self,
        n: int,
        sv: SamplingVariables,
        sa: SamplingAttributes,
        record: bool,
        exact_calc_interval: int,
    ) -> Tuple[SamplingVariables, SamplingAttributes]:
        for i in tqdm(range(n)):
            sv, sa = self._sample_indicator(sv=sv, sa=sa)
            sv, sa = self._sample_sigma_noise(sv=sv, sa=sa)
            sv, sa = self._sample_sigma_coef(sv=sv, sa=sa)

            if (i + 1) % exact_calc_interval == 0:
                sa = self._calculate_exact_sampling_attributes(X=self.X, sv=sv)

            if record:
                self._record_sequence(sv=sv, sa=sa)

        return sv, sa

    def sample(
        self,
        n_burn_in: int,
        n_sampling: int,
        random_state: Optional[int] = None,
        exact_calc_interval: int = 100,
    ):
        """
        MCMC sampling
        """
        np.random.seed(random_state)

        sv, sa = self._init_sampling()

        # Burn-in
        sv, sa = self._sample(
            n=n_burn_in,
            sv=sv,
            sa=sa,
            record=False,
            exact_calc_interval=exact_calc_interval,
        )

        # Sampling
        sv, sa = self._sample(
            n=n_sampling,
            sv=sv,
            sa=sa,
            record=True,
            exact_calc_interval=exact_calc_interval,
        )

        self.feature_posteriors_ = np.mean(self.indicator_sequence_, axis=0).tolist()
