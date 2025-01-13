from setuptools import setup, find_packages
from codecs import open

setup(
    name="autopocket",
    version="0.9.0",
    description="Automated Machine Learning project for financial data analysis.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="NC, FL, MM, MR, KT",
    author_email="flangiewicz@gmail.com",
    license="MIT",
    url="https://github.com/UserKrzysztof/AutoML-projekt-2",
    packages=find_packages(),
    install_requires=open("requirements.txt").readlines(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    keywords=[
        "automated machine learning",
        "automl",
        "machine learning",
        "data science",
        "data mining",
        "autopocket",
        "random forest",
        "decision tree",
        "linear model",
        "features selection",
        "features engineering",
        "economic data",
    ],
)