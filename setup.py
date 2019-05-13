import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seqprop",
    version="0.1",
    author="Johannes Linder",
    author_email="johannes.linder@hotmail.com",
    description="Stochastic Sequence Backpropagation",
    long_description=long_description,
    url="https://github.com/johli/seqprop",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
