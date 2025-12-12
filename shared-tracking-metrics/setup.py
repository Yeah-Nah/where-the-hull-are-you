"""Setup file for shared tracking metrics package."""

from setuptools import find_packages, setup

setup(
    name="tracking-metrics",
    version="0.1.0",
    description="Shared tracking metrics for boat tracking projects",
    author="Your Name",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "opencv-python",
        "mlflow",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "ruff",
        ],
    },
)
