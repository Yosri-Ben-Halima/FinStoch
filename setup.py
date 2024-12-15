from setuptools import setup, find_packages

# Read the requirements from the file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="FinStoch",  # Library name
    version="0.1.0",  # Initial version
    author="Yosri Ben Halima",  # Your name
    author_email="yosri.benhalima@ept.ucar.tn",  # Your email
    description="A library for stochastic processes in financial modeling.",
    long_description=open("README.md").read(),  # Ensure you have a README.md
    long_description_content_type="text/markdown",
    url="https://github.com/Yosri-Ben-Halima/FinStoch",  # Replace with your repo URL
    packages=find_packages(include=["FinStoch", "FinStoch.*"]),  # Automatically find packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Choose your license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Specify your Python version compatibility
    install_requires=requirements,  # Load requirements from requirements.txt
)
