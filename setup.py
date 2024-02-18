import setuptools


with open("README.md", 'r' , encoding = 'utf-8') as file:
    long_drescription = file.read()

__version__ = "0.0.0"

# AUTOHR_USER_NAME  = 'AhsanBilal7'
# AUTOHR_EMAIL  = 'abilal.bee20seecs@seecs.edu.pk'
# REPO_NAME  = 'End-to-End-DL-Project'
SRC_REPO  = 'Clip'

setuptools.setup(
    name = f"{SRC_REPO}",
    version = __version__,
    long_description_content = "text/markdown",
    package_dir={"": "src"},
    packages = setuptools.find_packages(where="src")
)