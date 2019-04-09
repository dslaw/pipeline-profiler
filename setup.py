from pathlib import Path
from setuptools import setup


setup(
    name="pipeline_profiler",
    version="0.1.0",
    summary="Add memory and wall-clock profiling to scikit-learn pipelines",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    license="LICENSE",
    install_requires=["memory_profiler"],
    py_modules=["pipeline_profiler"],
)
