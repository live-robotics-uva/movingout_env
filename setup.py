import os
from setuptools import find_packages, setup

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DESCRIPTION = "Moving Out is a benchmark suite for Physically-grounded Human-AI Collaboration."

# No longer need this hardcoded list
# CORE_REQUIREMENTS = [
#     "absl-py",
#     "gym",
#     "numpy",
#     "pygame",
#     "pyglet",
#     # "pymunk~=6.8.1",
#     "Pillow",
# ]

TEST_REQUIREMENTS = []
DEV_REQUIREMENTS = []

def get_requirements(file_path):
    """Reads the requirements from a given file."""
    with open(os.path.join(THIS_DIR, file_path), "r") as f:
        return [
            line.strip()
            for line in f
            if not line.strip().startswith("#") and line.strip()
        ]

def get_version():
    """Gets the version of the package."""
    locals_dict = {}
    with open(os.path.join(THIS_DIR, "moving_out", "version.py"), "r") as fp:
        exec(fp.read(), globals(), locals_dict)
    return locals_dict["__version__"]


setup(
    name="moving-out",
    version=get_version(),
    author="Xuhui Kang",
    license="ISC",
    description=DESCRIPTION,
    python_requires=">=3.8",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    # Read requirements from the requirements.txt file
    install_requires=get_requirements("requirements.txt"),
    extras_require={
        "dev": [*DEV_REQUIREMENTS, *TEST_REQUIREMENTS],
        "test": TEST_REQUIREMENTS,
    }
)