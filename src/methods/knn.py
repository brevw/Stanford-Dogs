import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind =task_kind

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        self.training_data = training_data
        self.training_labels = training_labels
        pred_labels = self.predict(training_data)
        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        def k_nn(example):
            distances = np.sqrt(((self.training_data - example.reshape(-1)) ** 2).sum(axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nn_labels  = self.training_labels[k_indices]
            if (self.task_kind == "classification"):
                #compute dominant class
                return np.argmax(np.bincount(k_nn_labels))
            else:
                #compute weighted average based on average labels 
                return np.mean(k_nn_labels, axis=0)
            
        test_labels = np.apply_along_axis(func1d=k_nn, axis=1, arr=test_data)
        return test_labels