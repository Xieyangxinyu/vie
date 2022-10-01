import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vie",
    version="0.0.1",
    author="Wenying Deng",
    author_email="wdeng5120@gmail.com",
    description="A package to estimate variable importance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wdeng5120/vie",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)