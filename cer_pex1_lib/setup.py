from setuptools import setup, find_packages

setup(
    name="cer_ex1_lib",
    version="1.0.0",
    description="Library for the first programming exercise of the CER lecture.",
    author="Tim Schneider",
    author_email="schneider@ias.informatik.tu-darmstadt.de",
    packages=find_packages(),
    python_requires=">=3.6.0",
    setup_requires=['wheel'],
    install_requires=[
        "numpy",
        "matplotlib",
        "jupyter"
    ]
)
