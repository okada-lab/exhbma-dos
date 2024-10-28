import numpy as np
import pytest

from exhbma.sampling import InversePrior


def test_inverse_prior_inside_value():
    norm = np.log(10)
    prior = InversePrior(low=1e-1, high=1)
    # Use pytest.approx since log calculation introduces an error.
    assert prior.prob(0.5) == pytest.approx(1 / (norm * 0.5))


def test_inverse_prior_edge_values():
    norm = np.log(10)
    prior = InversePrior(low=1e-1, high=1)
    assert prior.prob(0.1) == pytest.approx(1 / (norm * 0.1))
    assert prior.prob(1) == pytest.approx(1 / norm)


def test_inverse_prior_external_values():
    prior = InversePrior(low=1e-1, high=1)
    assert prior.prob(0.01) == 0
    assert prior.prob(2) == 0


def test_inverse_prior_negative_value():
    prior = InversePrior(low=1e-1, high=1)
    with pytest.raises(AssertionError) as e:
        prior.prob(-1)
    assert str(e.value) == "x must be greater than or equal to zero."
