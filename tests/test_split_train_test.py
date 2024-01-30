# tests/test_split_train_test.py
import unittest
from src.data.split_train_test import split_data
import pandas as pd
import numpy as np

class TestSplitTrainTest(unittest.TestCase):

    def setUp(self):
        # Create a dummy dataset
        self.X = pd.DataFrame(np.random.rand(100, 10))  # 100 samples, 10 features
        self.Y = pd.DataFrame(np.random.rand(100, 1))   # 100 samples, 1 target

    def test_split_data_proportions(self):
        X_train, Y_train, _, _, X_test, Y_test, _ = split_data(self.X, self.Y, train_prc=80, test_prc=20)
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)

    # Add more tests for stratified splits, random seed effect, etc.

if __name__ == '__main__':
    unittest.main()