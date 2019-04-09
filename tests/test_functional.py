from mprof import read_mprofile_file
from numpy import ones_like
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.dummy import DummyClassifier
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import pytest

from pipeline_profiler import profile


@pytest.fixture
def transformer():
    return StandardScaler()


@pytest.fixture
def predictor():
    return DummyClassifier(strategy="stratified")


class TestPipelineMethods(object):
    def test_fit(self, buffer, data, transformer):
        X, y = data
        pipeline = make_pipeline(transformer)

        with profile(pipeline, buffer):
            pipeline.fit(X, y)

            written = buffer.getvalue()
            assert ".fit" in written

    def test_transform(self, buffer, data, transformer):
        X, y = data
        pipeline = make_pipeline(transformer)

        with profile(pipeline, buffer):
            pipeline.fit(X, y)
            Xt = pipeline.transform(X)
            Xr = pipeline.inverse_transform(Xt)

            assert_allclose(Xr, X)
            written = buffer.getvalue()
            assert ".transform" in written
            assert ".inverse_transform" in written

    def test_predict(self, buffer, data, transformer, predictor):
        X, y = data
        pipeline = make_pipeline(transformer, predictor)

        with profile(pipeline, buffer):
            pipeline.fit(X, y)
            y_pred = pipeline.predict(X)

            assert y_pred.shape == y.shape
            written = buffer.getvalue()
            assert ".predict" in written

    def test_predict_proba(self, buffer, data, transformer, predictor):
        X, y = data
        pipeline = make_pipeline(transformer, predictor)

        with profile(pipeline, buffer):
            pipeline.fit(X, y)
            y_pred_proba = pipeline.predict_proba(X)

            assert y_pred_proba.shape == (len(y), 2)
            written = buffer.getvalue()
            assert ".predict_proba" in written


@pytest.fixture
def pipeline(transformer):
    predictor = DummyClassifier(strategy="constant", constant=1)
    pipeline = make_pipeline(transformer, predictor)
    return pipeline


class TestPipelineUsability(object):
    def test_pipeline_usable(self, buffer, data, pipeline):
        """Test that the fitted pipeline object can be used after patching."""

        X, y = data
        expected = ones_like(y)

        with profile(pipeline, buffer):
            pipeline.fit(X, y)

        actual = pipeline.predict(X)
        assert_array_equal(actual, expected)

    def test_pipeline_serializable(self, buffer, data, pipeline):
        """Test that the fitted pipeline object can be serialized."""

        X, y = data
        expected = ones_like(y)

        with profile(pipeline, buffer):
            pipeline.fit(X, y)

        with BytesIO() as model_buffer:
            joblib.dump(pipeline, model_buffer)
            model_buffer.seek(0)
            restored = joblib.load(model_buffer)

        actual = restored.predict(X)
        assert_array_equal(actual, expected)


def test_mprof_compatible_output(tmpdir, data, pipeline):
    X, y = data
    tmpfile = str(tmpdir / "test.dat")
    with open(tmpfile, "w") as fh, profile(pipeline, fh):
        pipeline.fit(X, y)

    contents = read_mprofile_file(tmpfile)
    assert contents["mem_usage"]
    assert contents["timestamp"]
    assert contents["func_timestamp"]
