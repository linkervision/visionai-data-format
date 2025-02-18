from setuptools import find_packages, setup

AUTHOR = "LinkerVision"
PACKAGE_NAME = "visionai-data-format"
PACKAGE_VERSION = "2.0.0"
DESC = "converter tool for visionai format"
REQUIRED = ["pydantic==2.*", "pillow"]
REQUIRES_PYTHON = ">=3.8, <4"
EXTRAS = {
    "test": [
        "pytest",
        "mock",
        "pre-commit",
    ],
}
with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()
setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    url="",
    description=DESC,
    author=AUTHOR,
    packages=find_packages(exclude=("tests")),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    python_requires=REQUIRES_PYTHON,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
