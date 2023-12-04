import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

class AdvertisingAnalysis:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def load_data(self):
        """
        Load the dataset from the specified file.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        return self.df

    def data_cleaning(self):
        """
        Check for null values in the dataset.
        """
        self.df.isnull().sum()

    def data_distribution(self):
        """
        Visualize the pair-wise relationships in the dataset.
        """
        sns.pairplot(self.df)
        plt.show()

    def check_linearity(self):
        """
        Visualize the relationship between advertising channels (TV, Radio, Newspaper) and Sales.
        """
        sns.pairplot(self.df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', aspect=1, kind='reg')

    def train_linear_regression(self, features, target):
        """
        Train a linear regression model.

        Args:
            features (list): List of feature column names.
            target (str): Target variable column name.

        Returns:
            LinearRegression: The trained linear regression model.
        """
        X = self.df[features]
        y = self.df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)
        regressor = LinearRegression(fit_intercept=True)
        regressor.fit(X_train, y_train)
        return regressor

    def predict(self, regressor, X_test):
        """
        Make predictions using a trained linear regression model.

        Args:
            regressor (LinearRegression): The trained linear regression model.
            X_test (pd.DataFrame): The test set features.

        Returns:
            np.ndarray: The predicted values.
        """
        return regressor.predict(X_test)

    def print_model_summary(self, features, target):
        """
        Print the summary statistics of the linear regression model.

        Args:
            features (list): List of feature column names.
            target (str): Target variable column name.
        """
        X = self.df[features]
        y = self.df[target]

        X2 = sm.add_constant(X)
        est = sm.OLS(y, X2)
        est2 = est.fit()
        print(est2.summary())




if __name__ == "__main__":
    analysis = AdvertisingAnalysis("Advertising.csv")
    
    data = analysis.load_data()
    analysis.data_cleaning()
    analysis.data_distribution()
    analysis.check_linearity()

    features_to_train = ['TV', 'Radio', 'Newspaper']
    target_variable = 'Sales'
    
    regressor = analysis.train_linear_regression(features_to_train, target_variable)
    
    X_test_example = pd.DataFrame({'TV': [200], 'Radio': [20], 'Newspaper': [10]})
    predictions = analysis.predict(regressor, X_test_example)
    
    print("Predictions:", predictions)
    
    analysis.print_model_summary(features_to_train, target_variable)
