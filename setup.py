# ---INFO-----------------------------------------------------------------------
# Author: Aditya Prakash
# Last edited: 18-12-2023

# ---DEPENDENCIES---------------------------------------------------------------
from setuptools import setup, find_packages

# ---METADATA-------------------------------------------------------------------
NAME = "dreamwalker"
DESCRIPTION = "Visual stimuli reconstruction using subject EEG."
URL = "https://github.com/adityaprakash-work/.git"
EMAIL = "adityaprakash.work@gmail.com"
AUTHOR = "Aditya Prakash"
REQUIRES_PYTHON = ">=3.9.0"
VERSION = "0.5.0"
REQUIRED = ["tqdm"]
EXTRAS = {}

# ---SETUP DREAMWALKER----------------------------------------------------------
setup(
    name="dreamwalker",
    version="0.1.0",
    author="Aditya Prakash",
    author_email="adityaprakash.work@gmail.com",
    description="Dreamwalker",
    long_description="Dreamwalker",
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
# ---END------------------------------------------------------------------------
