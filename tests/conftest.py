from io import StringIO
import numpy as np
import pytest


@pytest.fixture
def data():
    n_samples, n_features = (100, 2)
    rs = np.random.RandomState(13)
    X = rs.normal(size=(n_samples, n_features))
    y = rs.choice([0, 1], size=n_samples)
    return X, y


@pytest.fixture
def buffer():
    with StringIO() as buf:
        yield buf
