"""Setup file for model training package."""

from setuptools import find_packages, setup

setup(
    name="model-training",
    version="0.1.0",
    description="Train and evaluate custom maritime object detection models",
    author="Your Name",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "tracking-metrics @ file:../shared-tracking-metrics",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "mlflow>=2.8.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov",
            "ruff>=0.1.0",
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
        ],
    },
)
