from setuptools import find_packages, setup
# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='adversarial-test',
    packages=find_packages(),
    version='0.1.3',
    description='Adversarial test for tabular data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='TuHM',
    install_requires=[
        "numpy",
        "scikit-learn",
        "pandas",
        "catboost"
    ],
    tests_require=[
        "unittest",
    ],
    project_urls={
        "GitHub": "https://github.com/hmtu14/adversarial-test"
    }
    # license='MIT',
)