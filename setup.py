import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "MediLeaf_AI"
AUTHOR_USER_NAME = "surajkarki66"
SRC_REPO = "MediLeaf_AI"
AUTHOR_EMAIL = "suraj.karki500@protonmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description=" Medileaf is an application whose motive is to help the individual to identify medicinal plant with their properties by just scanning the leaf of any plant which might result creating curiosity about plant that lead to the preservation of the valuable plants as well as source of income.",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
