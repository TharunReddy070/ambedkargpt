"""Setup configuration for ambedkargpt package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ambedkargpt",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="SEMRAG: Semantic Graph-based RAG system for Dr. B.R. Ambedkar's works",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TharunReddy070/SEMRAG",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ambedkargpt=pipeline.ambedkargpt:main",
        ],
    },
)
