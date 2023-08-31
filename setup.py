"""Package installer."""
import os
from setuptools import setup
from setuptools import find_packages

LONG_DESCRIPTION = ""
if os.path.exists("README.md"):
    with open("README.md") as fp:
        LONG_DESCRIPTION = fp.read()

setup(
    name="featout",
    version="0.0.1",
    description="Removing features to avoid shortcut learning",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Nina Wiedemann",
    author_email=("nwiedemann@ethz.ch"),
    license="MIT",
    url="https://github.com/NinaWie/featout",
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "torchvision",
        "captum",
        "scipy",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages("."),
    python_requires=">=3.8",
    scripts=[],
)
