# Adversarial test: simple way to know if your train data and test data are similar

We combine our train and test data, labeling them 0 for the training data and 1 for the test data, mix them up, then see if we are able to correctly re-identify them using a binary classifier.

If a classifier can identify whether a sample comes from train or test data set, we know that there's at least one feature in your data is shifted; use feature importance methods to point out the shifted feature(s)

# Get Started and Documentation
To install from pip:
```
pip install adversarial-test
```

Code example:
- [Using adversarial test with category features]()
- See more usages in [notebooks]() directory
