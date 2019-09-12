import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stablab",
    version="1.0",
    author="Blake Barker, Jeffrey Humpherys, Joshua Lytle, Jalen Morgan, and Taylor Paskett",
    author_email="blake@mathematics.byu.edu",
    description="Package for examining the stability of traveling waves using"+
    " a variety of numerical methods.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nonlinear-waves/stablab_python",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
