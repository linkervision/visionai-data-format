from setuptools import find_packages, setup

AUTHOR = "LinkerVision"
PACKAGE_NAME = "visionai-data-format"
PACKAGE_VERSION = "0.1.2"
DESC = "converter tool for visionai format"

setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    url="",
    description=DESC,
    author=AUTHOR,
    packages=find_packages(),
    install_requires=["pydantic"],
    python_requires=">=3.9, <4",
)
