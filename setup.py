from setuptools import find_packages, setup

setup(
    name='adversarial-test',
    packages=find_packages(),
    version='0.1.1',
    description='Adversarial test for tabular data',
    author='TuHM',
    install_requires=[
        "numpy",
        "scikit-learn",
        "pandas",
        "catboost"
    ],
    tests_require=[
        "unittest",
    ]
    # license='MIT',
)