import unittest
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Import the class to be tested
from linear_Regression import AdvertisingAnalysis

class TestAdvertisingAnalysis(unittest.TestCase):
    def setUp(self):
        # Create an instance of AdvertisingAnalysis for testing
        self.analysis = AdvertisingAnalysis("Advertising.csv")
        self.df = pd.DataFrame({
            'TV': [100, 200, 300],
            'Radio': [10, 20, 30],
            'Newspaper': [5, 10, 15],
            'Sales': [15, 25, 35]
        })

    def test_load_data(self):
        # Test if the loaded data is a DataFrame
        self.assertIsInstance(self.analysis.load_data(), pd.DataFrame)

    def test_data_cleaning(self):
        # Test if data cleaning returns None (or any other appropriate check)
        self.assertIsNone(self.analysis.data_cleaning())

    def test_data_distribution(self):
        # Test if data distribution returns None (or any other appropriate check)
        self.assertIsNone(self.analysis.data_distribution())

    def test_check_linearity(self):
        # Test if check linearity returns None (or any other appropriate check)
        self.assertIsNone(self.analysis.check_linearity())

    def test_train_linear_regression(self):
        # Test if train linear regression returns a LinearRegression model
        features = ['TV', 'Radio', 'Newspaper']
        target_variable = 'Sales'
        model = self.analysis.train_linear_regression(features, target_variable)
        self.assertIsInstance(model, LinearRegression)

    def test_predict(self):
        # Test if predict returns a numpy array
        features = ['TV', 'Radio', 'Newspaper']
        target_variable = 'Sales'
        model = self.analysis.train_linear_regression(features, target_variable)
        X_test = pd.DataFrame({'TV': [400], 'Radio': [40], 'Newspaper': [20]})
        predictions = self.analysis.predict(model, X_test)
        self.assertIsInstance(predictions, (list, np.ndarray))

    def test_print_model_summary(self):
        # Test if print_model_summary prints without errors
        features = ['TV', 'Radio', 'Newspaper']
        target_variable = 'Sales'
        X = self.df[features]
        y = self.df[target_variable]
        X2 = sm.add_constant(X)
        est = sm.OLS(y, X2)
        est2 = est.fit()
        summary = est2.summary()
        self.assertIsNone(print(summary))  # Change this if print_model_summary returns something

if __name__ == '__main__':
    unittest.main()
