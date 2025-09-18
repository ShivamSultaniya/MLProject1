"""
Setup script for Multi-Modal Concentration Analysis System
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="concentration-analysis-system",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-modal concentration analysis system using computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/concentration-analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Multimedia :: Video :: Capture",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=0.971",
            "pre-commit>=2.20",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-autodoc-typehints>=1.19",
        ],
        "gpu": [
            "torch>=1.12.0+cu116",
            "torchvision>=0.13.0+cu116",
        ],
    },
    entry_points={
        "console_scripts": [
            "concentration-demo=demo_realtime:main",
            "concentration-train=src.training.train_all:main",
            "concentration-eval=src.evaluation.evaluate_system:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
    zip_safe=False,
)
