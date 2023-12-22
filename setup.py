# ---INFO-----------------------------------------------------------------------
# Author: Aditya Prakash
# Last edited: 19-12-2023

# ---DEPENDENCIES---------------------------------------------------------------
from setuptools import setup, find_packages

# ---METADATA-------------------------------------------------------------------
NAME = "dreamwalker"
DESCRIPTION = "Visual stimuli reconstruction using subject EEG."
URL = "https://github.com/adityaprakash-work/DreamWalker.git"
EMAIL = "adityaprakash.work@gmail.com"
AUTHOR = "Aditya Prakash"
REQUIRES_PYTHON = ">=3.9.0"
VERSION = "0.1.0"
REQUIRED = [
    "opendatasets",
    "tqdm",
    "torch",
    "torchvision",
    "mne",
    "numpy",
    "Pillow",
    "tensorboard",
    "scikit-learn",
]

# pytorch-generative is being provided internally until this package comes out
# of nascency. It's hosted on GitHub, but doesn't have a setup.py file. There
# is a requirements.txt file however, added in 'REQUIRED'.

pytorch_generative_url = "https://github.com/EugenHotaj/pytorch-generative.git"
REQUIRED += ["git+pytorch_generative_url@master#egg=pytorch_generative"]

EXTRAS = {}

# ---SETUP DREAMWALKER----------------------------------------------------------
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9.0",
    ],
)

# ---END------------------------------------------------------------------------
