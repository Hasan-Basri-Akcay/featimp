import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="featimp",
    version="v0.1.7",
    author="Hasan Basri Akcay",
    author_email="hasan.basri.akcay@gmail.com",
    description="Feature importance for machine learning",
    long_description=(
        "Featimp helps with feature understanding, "
        "calculating feature importances, feature "
        "debugging, and leakage detection"
    ),
    long_description_content_type="text/markdown",
    url="https://github.com/Hasan-Basri-Akcay/featimp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["pandas", "numpy", "matplotlib", "seaborn", 
                      "scikit-learn", "lightgbm", "catboost", "scipy"],
    keywords=["python", "data science", "data analysis", "exploratory data analysis", 
              "feature importance", "beginner"],
)
