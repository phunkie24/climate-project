from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="climate-rainfall-analysis",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Machine Learning for Climate Pattern Detection in Sub-Saharan Africa",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/climate-rainfall-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "climate-train=scripts.train_model:main",
            "climate-predict=scripts.generate_predictions:main",
            "climate-dashboard=scripts.dashboard:main",
        ],
    },
)
