from setuptools import find_packages, setup

AUTHOR = "LinkerVision"
PACKAGE_NAME = "visionai-data-format"
PACKAGE_VERSION = "0.1.3"
DESC = "converter tool for visionai format"
REQUIRED = ["pydantic"]
REQUIRES_PYTHON = ">=3.7, <4"
EXTRAS = {
    "test": [
        "pytest",
        "mock",
        "pre-commit",
    ],
}
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
)
