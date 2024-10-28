from typing import List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from tqdm.auto import tqdm


class SamplingVariables(BaseModel):
    indicator_index: List[int] = Field(
        ..., description="First es_k indexes are non-zero and others are zero."
    )


class SamplingAttributes(BaseModel):
    log_likelihood: float = Field(..., description="Log-likelihood of the model.")


class Replica(BaseModel):
    sv: SamplingVariables = Field(..., description="Sampling variables.")
    sa: SamplingAttributes = Field(..., description="Sampling attributes.")


class AESCVE(object):
    """Cross validation error sampling using the AES method."""

    def __init__(
        self,
        X,
        y,
        beta_list: List[float],
        es_k: int,
        model: LinearRegression,
        init_indicator: Optional[List[List[int]]] = None,
    ):
        """Initialize the AESBFE object."""
        self.X = X
        self.y = y
        self.beta_list = beta_list
        self.es_k = es_k
        self.model = model

        self.n_replica = len(beta_list)
        self.n_features = X.shape[1]

        self.init_indicator = (
            self._sample_indicator_from_prior()
            if init_indicator is None
            else init_indicator
        )

        self.indicator_sequence_: List[List[List[int]]] = [
            [] for _ in range(self.n_replica)
        ]
        self.log_likelihood_sequence_: List[List[float]] = [
            [] for _ in range(self.n_replica)
        ]

        self.exchange_count_: List[int] = [0] * self.n_replica
        self.acceptance_count_: List[int] = [0] * self.n_replica

    def _sample_indicator_from_prior(self) -> List[List[int]]:
        indicator_list = []
        for _ in range(self.n_replica):
            indicator_list.append(
                np.random.choice(
                    self.n_features, self.n_features, replace=False
                ).tolist()
            )
        return indicator_list

    def _calc_mse(self, y_true, y_pred) -> float:
        mse = np.mean((y_true - y_pred) ** 2)
        return mse

    def _calculate_log_likelihood(
        self, indicator_index: List[int], k_fold: int = 10
    ) -> float:
        # Cross Validation Error based log-likelihood
        X_ = self.X[:, indicator_index]

        splitter = KFold(n_splits=k_fold, shuffle=True)
        mse_scores = []
        for train_index, test_index in splitter.split(X_):
            X_train, X_test = X_[train_index], X_[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)
            mse_scores.append(self._calc_mse(y_test, y_pred))

        log_likelihood = -np.mean(mse_scores)
        return log_likelihood

    def _sample_indicator(self, beta: float, replica: Replica) -> Tuple[Replica, bool]:
        rm_index = np.random.choice(self.es_k)
        add_index = np.random.choice(self.n_features - self.es_k) + self.es_k
        replica.sv.indicator_index[rm_index], replica.sv.indicator_index[add_index] = (
            replica.sv.indicator_index[add_index],
            replica.sv.indicator_index[rm_index],
        )

        new_log_likelihood = self._calculate_log_likelihood(
            indicator_index=replica.sv.indicator_index[: self.es_k]
        )

        accepted = False
        if np.log(np.random.rand()) < beta * (
            new_log_likelihood - replica.sa.log_likelihood
        ):
            replica.sa.log_likelihood = new_log_likelihood
            accepted = True
        else:
            (
                replica.sv.indicator_index[rm_index],
                replica.sv.indicator_index[add_index],
            ) = (
                replica.sv.indicator_index[add_index],
                replica.sv.indicator_index[rm_index],
            )
        return replica, accepted

    def _exchange(self, replicas: List[Replica], start: int = 0) -> List[Replica]:
        """Exchange replicas."""
        for i in range(start, self.n_replica - 1, 2):
            beta1, replica1 = self.beta_list[i], replicas[i]
            beta2, replica2 = self.beta_list[i + 1], replicas[i + 1]

            d_beta = beta2 - beta1
            d_log_likelihood = replica2.sa.log_likelihood - replica1.sa.log_likelihood

            if np.log(np.random.rand()) < -d_log_likelihood * d_beta:
                replicas[i], replicas[i + 1] = replicas[i + 1], replicas[i]
                self.exchange_count_[i] += 1
                self.exchange_count_[i + 1] += 1
        return replicas

    def _record_sequence(self, replicas: List[Replica]):
        for i in range(self.n_replica):
            self.indicator_sequence_[i].append(
                replicas[i].sv.indicator_index[: self.es_k]
            )
            self.log_likelihood_sequence_[i].append(replicas[i].sa.log_likelihood)

    def _sample(
        self,
        n: int,
        replicas: List[Replica],
        record: bool,
    ) -> List[Replica]:
        for i in tqdm(range(n)):
            for j in range(self.n_replica):
                replicas[j], accepted = self._sample_indicator(
                    beta=self.beta_list[j], replica=replicas[j]
                )
                if accepted:
                    self.acceptance_count_[j] += 1

            replicas = self._exchange(replicas=replicas, start=i % 2)

            if record:
                self._record_sequence(replicas=replicas)
        return replicas

    def _init_sampling(self) -> List[Replica]:
        replicas: List[Replica] = []
        for i in range(self.n_replica):
            sv = SamplingVariables(indicator_index=self.init_indicator[i])
            log_likelihood = self._calculate_log_likelihood(
                indicator_index=sv.indicator_index[: self.es_k]
            )
            sa = SamplingAttributes(log_likelihood=log_likelihood)
            replicas.append(Replica(sv=sv, sa=sa))
        return replicas

    def sample(
        self,
        n_burn_in: int,
        n_sampling: int,
        random_state: Optional[int] = None,
        n_dos_bins: int = 100,
    ):
        np.random.seed(random_state)

        replicas: List[Replica] = self._init_sampling()

        # Burn-in
        replicas = self._sample(
            n=n_burn_in,
            replicas=replicas,
            record=False,
        )

        # Sampling
        replicas = self._sample(
            n=n_sampling,
            replicas=replicas,
            record=True,
        )

        # Post processing
        self.dos_, self.bin_positions_, self.z_const_ = self._estimate_dos(
            n_bins=n_dos_bins
        )

    def _estimate_dos(
        self,
        n_bins: int = 100,
        tolerance: float = 1e-6,
        max_iter: int = 10 ** 4,
        ll_threshold: float = -0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_log_likelihood = []
        n_samples_list = []
        for i in range(self.n_replica):
            np_log_likelihood = np.array(self.log_likelihood_sequence_[i])
            index = np_log_likelihood > ll_threshold
            all_log_likelihood.extend(np_log_likelihood[index].tolist())
            n_samples_list.append(np.sum(index))

        hist, bin_edges = np.histogram(all_log_likelihood, bins=n_bins)
        bin_positions = (bin_edges[:-1] + bin_edges[1:]) / 2

        z_const = np.ones(self.n_replica)
        for i in range(max_iter):
            prev_z_const = z_const.copy()

            dos = hist / (
                np.dot(
                    np.exp(np.outer(bin_positions, self.beta_list)),
                    np.array(n_samples_list) / z_const,
                )
            )
            z_const = np.dot(np.exp(np.outer(self.beta_list, bin_positions)), dos)

            err = np.mean(np.abs(z_const - prev_z_const))
            if err < tolerance:
                break
        return dos, bin_positions, z_const
