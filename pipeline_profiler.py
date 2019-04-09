"""Profile scikit-learn Pipelines."""

from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from memory_profiler import memory_usage


# NB: If the interval is above 1e-6, the memory profiler may execute the
#     function it is profiling multiple times. See:
#     https://github.com/pythonprofilers/memory_profiler/issues/216
INTERVAL = 1e-6 - 1
METHODS = (
    "fit",
    "transform",
    "inverse_transform",
    "predict",
    "predict_proba",
    "predict_log_proba",
    "decision_function",
    "fit_transform",
    "fit_predict",
)


def format_memline(line):
    """Create the output string for a memory profile recording."""
    return "MEM {0:.6f} {1:.4f}\n".format(*line)


def format_funcline(name, mem_usage):
    """Create the output string for a function profile interval."""
    start_memory, start_timestamp = mem_usage[0]
    end_memory, end_timestamp = mem_usage[-1]
    return "FUNC {name} {0:.6f} {1:.4f} {2:.6f} {3:.4f}\n".format(
        start_memory,
        start_timestamp,
        end_memory,
        end_timestamp,
        name=name,
    )


def mprofile(fn, fh):
    def inner(*args, **kwargs):
        mem_usage, ret = memory_usage(
            (fn, args, kwargs),
            max_usage=False,
            retval=True,
            timestamps=True,
            # TODO: check if this will get used when `proc` is a tuple.
            multiprocess=True,
            interval=INTERVAL,
        )

        for line in mem_usage:
            fh.write(format_memline(line))
        fh.write(format_funcline(fn.__qualname__, mem_usage))

        return ret
    return inner


def iter_methods(pipeline):
    """Iterate over the estimator API methods for each estimator."""
    for name, estimator in pipeline.steps:
        for method_name in METHODS:
            method = getattr(estimator, method_name, None)
            if method is not None and callable(method):
                yield (name, estimator), (method_name, method)


def add_profiling(pipeline, fh, ctx):
    """Decorate estimator methods of each estimator in the pipeline."""

    for (name, estimator), (method_name, method) in iter_methods(pipeline):
        # Store original function pointer.
        ctx[name].update({method_name: method})

        # Replace with decorated version.
        decorated_method = mprofile(method, fh)
        setattr(estimator, method_name, decorated_method)

    return pipeline


def remove_profiling(pipeline, ctx):
    """Restore original estimator methods."""

    for (name, estimator), (method_name, method) in iter_methods(pipeline):
        # Replace decorated method with original.
        original = ctx[name][method_name]
        setattr(estimator, method_name, original)

    return pipeline


@contextmanager
def profile(pipeline, fh=None):
    """Add profiling to a scikit-learn Pipeline.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The pipeline to be profiled.
    fh : file-like, optional
        An open file handle for the profiling results to be
        written to. If None (default), a file will be created
        in the current directory with the prefix ``pprof_``.

    Returns
    -------
    patched : sklearn.pipeline.Pipeline
        The input pipeline with decorated estimator API methods.

    Examples
    --------
    >>> with io.StringIO() as buffer, profile(pipeline, buffer):
    ...     pipeline.fit(X, y)
    ...     y_pred = pipeline.predict(X)
    ...     profiling_results = buffer.getvalue()
    """

    context = defaultdict(dict)
    try:
        close_fh = fh is None
        if fh is None:
            ts = int(datetime.utcnow().timestamp())
            filename = "pprof_{}.dat".format(ts)
            fh = open(filename, "w")

        patched = add_profiling(pipeline, fh, context)
        yield patched
    finally:
        remove_profiling(pipeline, context)
        if close_fh:
            fh.close()
