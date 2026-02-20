from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="openengine",
    version="1.0.0",
    description="A backtesting and live trading engine library for Indian markets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="OpenEngine Contributors",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/openengine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.31.0",
        "duckdb>=0.9.0",
        "pandas>=2.0.0",
        "yfinance>=0.2.0"
    ],
    entry_points={
        'console_scripts': [
            'openengine=openengine.main:main'
        ]
    },
)
