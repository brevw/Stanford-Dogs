import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters


    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        C = get_n_classes(training_labels)
        D = training_data.shape[1]
        training_labels_one_hot = label_to_onehot(training_labels, C)
        weights = np.random.normal(0, 0.1, (D, C))
        for it in range(self.max_iters):
            #compute scores (we ensure numerical stability)
            scores = training_data @ weights
            scores -= np.max(scores, axis = 1, keepdims = True)
            scores = np.exp(scores)
            # compute probabilities (using soft_max)
            probabilities = scores / np.sum(scores, axis=1, keepdims=True)

            #debugging (to delete after)
            loss_reg = -np.sum(training_labels_one_hot * np.log(probabilities + 1e-12))/training_data.shape[0]
            print("loss regression = ", loss_reg)

            # compute gradient of loss cross entropy with respect to weights
            grad_logistic_reg = training_data.T @ (probabilities - training_labels_one_hot)

            # gradient step
            weights = weights - self.lr * grad_logistic_reg
        
        # capture weights after training
        self.weights = weights
        pred_labels = self.predict(training_data)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        #compute scores (we ensure numerical stability)
        scores = test_data @ self.weights
        scores -= np.max(scores, axis = 1, keepdims = True)
        scores = np.exp(scores)
        #compute probablities
        probabilities = scores / np.sum(scores, axis=1, keepdims=True)
        pred_labels = onehot_to_label(probabilities)
        return pred_labels
