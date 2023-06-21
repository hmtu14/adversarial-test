import unittest
import numpy as np
import pandas as pd
from adversarial_test import AdversarialModel

class TestAdversarialModel(unittest.TestCase):
    def setUp(self):
        self.model = AdversarialModel()

    def test_fit_same_distribution(self):
        df1 = pd.DataFrame(np.random.rand(1000, 50))
        df2 = pd.DataFrame(np.random.rand(1000, 50))
        self.model.fit(df1, df2)
        self.assertEqual(self.model.evaluate(metadata=False), "pass")
    
    def test_fit_diff_distribution(self):
        df1 = pd.DataFrame(np.random.rand(1000, 50))
        df2 = pd.DataFrame(np.random.randint(1, (1000, 50)))
        self.model.fit(df1, df2)
        self.assertEqual(self.model.evaluate(metadata=False), "fail")

    def test_fit_cat_features_same_distribution(self):
        df1 = pd.DataFrame(np.random.rand(1000, 50))
        string_list = ['apple', 'banana', 'cherry', 'date', 'elderberry']
        df1["cat1"] = np.random.choice(string_list, size=len(df1))

        df2 = pd.DataFrame(np.random.rand(1000, 50))
        string_list = ['apple', 'banana', 'cherry', 'date', 'elderberry']
        df2["cat1"] = np.random.choice(string_list, size=len(df2))

        self.model.fit(df1, df2, cat_features=["cat1"])
        self.assertEqual(self.model.evaluate(metadata=False), "pass")

    def test_fit_cat_features_diff_distribution(self):
        df1 = pd.DataFrame(np.random.rand(1000, 50))
        string_list = ['apple', 'banana', 'cherry', 'date', 'elderberry']
        df1["cat1"] = np.random.choice(string_list, size=len(df1))

        df2 = pd.DataFrame(np.random.rand(1000, 50))
        string_list = ['red', 'green', 'blue', 'orange', 'yellow']
        df2["cat1"] = np.random.choice(string_list, size=len(df2))

        self.model.fit(df1, df2, cat_features=["cat1"])
        self.assertEqual(self.model.evaluate(metadata=False), "fail")

    def test_fit_stratified_group_kfold_same_distribution(self):
        df1 = pd.DataFrame(np.random.rand(1000, 50))
        string_list = ['apple', 'banana', 'cherry', 'date', 'elderberry']
        df1["cat1"] = np.random.choice(string_list, size=len(df1))

        df2 = pd.DataFrame(np.random.rand(1000, 50))
        string_list = ['apple', 'banana', 'cherry', 'date', 'elderberry']
        df2["cat1"] = np.random.choice(string_list, size=len(df2))

        self.model.fit(df1, df2, cat_features=["cat1"], groups_col=["cat1"])
        self.assertEqual(self.model.evaluate(metadata=False), "pass")

    def test_fit_stratified_group_kfold_diff_distribution(self):
        df1 = pd.DataFrame(np.random.rand(1000, 50))
        string_list = ['apple', 'banana', 'cherry', 'date', 'elderberry']
        df1["cat1"] = np.random.choice(string_list, size=len(df1))
        string_list = ['apple', 'banana', 'cherry', 'date', 'elderberry']
        df1["cat2"] = np.random.choice(string_list, size=len(df1))

        df2 = pd.DataFrame(np.random.rand(1000, 50))
        string_list = ['apple', 'banana', 'cherry', 'date', 'elderberry']
        df2["cat1"] = np.random.choice(string_list, size=len(df2))
        string_list = ['red', 'green', 'blue', 'orange', 'yellow']
        df2["cat2"] = np.random.choice(string_list, size=len(df2))

        self.model.fit(df1, df2, cat_features=["cat1", "cat2"], groups_col=["cat1"])
        self.assertEqual(self.model.evaluate(metadata=False), "fail")
