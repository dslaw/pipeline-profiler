from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from pipeline_profiler import iter_methods, mprofile


def add(x, y):
    return x + y


class TestMPofile(object):
    def test_returns_value(self, buffer):
        wrapped = mprofile(add, buffer)
        assert wrapped(1, 1) == 2

    def test_writes_memory_usage(self, buffer):
        wrapped = mprofile(add, buffer)
        wrapped(1, 1)

        written = buffer.getvalue()
        assert any(line.startswith("MEM") for line in written.split("\n"))
        assert any(line.startswith("FUNC add") for line in written.split("\n"))

    def test_wrapped_called_once(self, buffer):
        # `memory_profiler` may call the function it is profiling
        # multiple times under certain conditions. Ensure we've
        # stopped that from happening.
        def fn(x):
            fn.n_calls = getattr(fn, "n_calls", 0) + 1
            return x

        wrapped = mprofile(fn, buffer)
        wrapped(10)

        assert fn.n_calls == 1


class FakeTransformer(BaseEstimator):
    def fit(self, X, y=None):
        return X

    def transform(self, X):
        return X

    def helper(self):
        pass


class FakeClassifier(BaseEstimator):
    def helper(self):
        pass

    def fit(self, X, y=None):
        return X

    def predict(self, X):
        return X

    def predict_proba(self, X):
        return X


class TestIterMethods(object):
    def test_iterates_steps(self):
        trf, clf = FakeTransformer(), FakeClassifier()
        pipeline = Pipeline([("trf", trf), ("clf", clf)])

        # There should be one (name, estimator) pair for
        # each discovered method.
        steps, _ = zip(*iter_methods(pipeline))
        assert steps == (
            ("trf", trf),
            ("trf", trf),
            ("clf", clf),
            ("clf", clf),
            ("clf", clf),
        )

    def test_iterates_methods(self):
        trf, clf = FakeTransformer(), FakeClassifier()
        pipeline = Pipeline([("trf", trf), ("clf", clf)])

        # Order is not important.
        _, methods = zip(*iter_methods(pipeline))
        assert set(methods) == {
            ("fit", trf.fit),
            ("transform", trf.transform),
            ("fit", clf.fit),
            ("predict", clf.predict),
            ("predict_proba", clf.predict_proba),
        }
