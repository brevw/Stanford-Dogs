import numpy as np
import sys
from utils import append_bias_term

class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    def __init__(self, lmda):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.lmda = lmda

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """
        ## my_impl starts here
        training_data_bias = append_bias_term(training_data)
        weigths = np.linalg.solve(training_data_bias.T @ training_data_bias, 
                                                   training_data_bias.T @ training_labels)
        self.weights = weigths
        pred_regression_targets = training_data_bias @ weigths
        ## my_impl stops here

        return pred_regression_targets


def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,regression_target_size)
        """
        ## my_impl starts here
        pred_regression_targets = test_data @ self.weights
        ## my_impl stops here

        return pred_regression_targets
