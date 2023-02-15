from setuptools import setup, find_packages

setup(
    name="cer_pex3_lib",
    version="1.0.0",
    description="Library for the third programming exercise of the CER lecture.",
    author="Tim Schneider",
    author_email="schneider@ias.informatik.tu-darmstadt.de",
    packages=find_packages(),
    python_requires=">=3.6.0",
    setup_requires=["wheel"],
    install_requires=[
        "numpy",
        "matplotlib",
        "jupyter"
    ],
    include_package_data=True,
    package_data={"": ["*.json"]},
)
