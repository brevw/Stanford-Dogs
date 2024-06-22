import argparse

import numpy as np

from matplotlib import pyplot as plt
from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression 
from src.methods.knn import KNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os
np.random.seed(100)

def KFold_cross_validation(X, Y, K, method_obj):
    '''
    K-Fold Cross validation function for K-NN

    Inputs:
        X : training data, shape (NxD)
        Y: training labels, shape (N,)
        K: number of folds (K in K-fold)
    Returns:
        Average validation accuracy.
    '''
    N = X.shape[0]

    if isinstance(method_obj, LogisticRegression) :
        compute = accuracy_fn
    elif isinstance(method_obj, LinearRegression):
        compute = mse_fn
    elif isinstance(method_obj, KNN) and method_obj.task_kind == "classification":
        compute = accuracy_fn
    else:
        compute = mse_fn
    
    accuraciesVal = []  # list of accuracies
    accuraciesTrain = []
    #Split the data into training and validation folds:
    all_ind = np.arange(N)
    #all the indices of the training dataset
    split_size = N // K
    for fold_ind in range(K):
        # Indices of the validation and training examples
        val_ind = all_ind[fold_ind * split_size : (fold_ind + 1) * split_size]
        train_ind = np.setdiff1d(all_ind, val_ind)
        
        X_train_fold = X[train_ind, :]
        Y_train_fold = Y[train_ind]
        X_val_fold = X[val_ind, :]
        Y_val_fold = Y[val_ind]

        # Run KNN using the data folds you found above.
        # YOUR CODE HERE
        Y_train_fold_pred = method_obj.fit(X_train_fold, Y_train_fold)
        Y_val_fold_pred = method_obj.predict(X_val_fold)
        acc_val = compute(Y_val_fold_pred, Y_val_fold)
        acc_train = compute(Y_train_fold_pred, Y_train_fold)
        accuraciesVal.append(acc_val)
        accuraciesTrain.append(acc_train)
    
    #Find the average validation accuracy over K:
    ave_acc_val = np.mean(accuraciesVal)
    ave_acc_train = np.mean(accuraciesTrain)
    return ave_acc_train, ave_acc_val

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors

    ##EXTRACTED FEATURES DATASET
    if args.data_type == "features":
        feature_data = np.load('features.npz',allow_pickle=True)
        xtrain, xtest, ytrain, ytest, ctrain, ctest =feature_data['xtrain'],feature_data['xtest'],\
        feature_data['ytrain'],feature_data['ytest'],feature_data['ctrain'],feature_data['ctest']

    ##ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path,'dog-small-64')
        xtrain, xtest, ytrain, ytest, ctrain, ctest = load_data(data_dir)

    ##TODO: ctrain and ctest are for regression task. (To be used for Linear Regression and KNN)
    ##TODO: xtrain, xtest, ytrain, ytest are for classification task. (To be used for Logistic Regression and KNN)



    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test and not(args.KFold_plot):
        N = xtrain.shape[0]
        validation_size = int(N * 0.2)
        rand_idx = np.random.permutation(N)
        val_idx = rand_idx[:validation_size]
        train_idx = rand_idx[validation_size:]
        xtest = xtrain[val_idx,:]
        ytest = ytrain[val_idx]
        ctest = ctrain[val_idx]
        xtrain = xtrain[train_idx,:]
        ytrain = ytrain[train_idx]
        ctrain = ctrain[train_idx]
        pass
    
    ### WRITE YOUR CODE HERE to do any other data processing
    mean = np.mean(xtrain, axis=0, keepdims=True)
    std = np.std(xtrain, axis=0, keepdims=True)
    xtrain = normalize_fn(xtrain, mean, std)
    xtest = normalize_fn(xtest, mean, std)
    xtrain = append_bias_term(xtrain)
    xtest = append_bias_term(xtest)

    # Initialize the booelans to plot the accuracy curves using KFold cross validation
    plotLinearCenter = False
    plotLogisticBreed = False
    plotKnnCenter = False
    plotKnnBreed = False

    

    ## 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "knn":
        if args.task == "center_locating":
            method_obj = KNN(args.K, "regression")
            plotKnnCenter = True
        else:
            method_obj = KNN(args.K, "classification")
            plotKnnBreed = True
    elif args.method == "linear_regression":
        method_obj = LinearRegression(args.lmda)
        plotLinearCenter = True
    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(args.lr, args.max_iters)
        plotLogisticBreed = True


    ## 4. Train and evaluate the method
    if (not(args.KFold_plot)):

        if args.task == "center_locating":
            # Fit parameters on training data
            preds_train = method_obj.fit(xtrain, ctrain)

            # Perform inference for training and test data
            train_pred = method_obj.predict(xtrain)
            preds = method_obj.predict(xtest)

            ## Report results: performance on train and valid/test sets
            train_loss = mse_fn(train_pred, ctrain)
            loss = mse_fn(preds, ctest)

            print(f"\nTrain loss = {train_loss:.3f}% - Test loss = {loss:.3f}")

        elif args.task == "breed_identifying":

            # Fit (:=train) the method on the training data for classification task
            preds_train = method_obj.fit(xtrain, ytrain)

            # Predict on unseen data
            preds = method_obj.predict(xtest)

            ## Report results: performance on train and valid/test sets
            acc = accuracy_fn(preds_train, ytrain)
            macrof1 = macrof1_fn(preds_train, ytrain)
            print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

            acc = accuracy_fn(preds, ytest)
            macrof1 = macrof1_fn(preds, ytest)
            print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
        else:
            raise Exception("Invalid choice of task! Only support center_locating and breed_identifying!")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.
    else:
       
        if plotLinearCenter:
            plt.figure()
            K = 5
            lmdaArray = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
            mse_train = []
            mse_val   = []
            method_obj = LinearRegression(0)
            for lmda in lmdaArray:
                method_obj.lmda = lmda
                mse_train_avg, mse_val_avg = KFold_cross_validation(xtrain, ctrain, K, method_obj)
                mse_train.append(mse_train_avg)
                mse_val.append(mse_val_avg)
            #plt.title(f"Impact of Regularization Strength (Lambda) on Linear Regression Performance")
            plt.loglog(lmdaArray, mse_val, 'ro-', label = "validation set")
            plt.loglog(lmdaArray, mse_train, 'bo-', label = "training set")
            plt.xlabel("lambda")
            plt.ylabel("mse")
            plt.legend()
            plt.show()
        elif plotLogisticBreed:
            plt.figure()
            color = ["r", "b", "g", "y", "k"]
            K = 5
            lr_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1]
            max_iters_grid = [200, 300, 400, 600, 800]
            accuracies_train = np.zeros((len(lr_grid), len(max_iters_grid)))
            accuracies_val = np.zeros((len(lr_grid), len(max_iters_grid)))
            method_obj = LogisticRegression(0, 0)
            for i, lr in enumerate(lr_grid):
                for j, max_iters in enumerate(max_iters_grid):
                    method_obj.lr = lr
                    method_obj.max_iters = max_iters
                    acc_train_avg, acc_val_avg = KFold_cross_validation(xtrain, ytrain, K, method_obj)
                    accuracies_train[i, j] = acc_train_avg
                    accuracies_val[i, j] = acc_val_avg
            for i, lr in enumerate(lr_grid):
                plt.plot(max_iters_grid, accuracies_val[i,:], "o-", color = color[i], label = f"lr = {lr}")
            plt.xlabel("max_iters")
            plt.ylabel("accuracy (%)")
            #plt.title("Impact of Learning Rate and Maximum Iterations on Model Accuracy")
            plt.legend()
            plt.show()
        elif plotKnnBreed:
            plt.figure()
            K = 5
            k_grid = [1, 5, 10, 15, 20, 30, 40, 60, 80]
            accuracies_train = []
            accuracies_val   = []
            method_obj = KNN(0, "classification")
            for k in k_grid:
                method_obj.k = k
                acc_train_avg, acc_val_avg = KFold_cross_validation(xtrain, ytrain, K, method_obj)
                accuracies_train.append(acc_train_avg)
                accuracies_val.append(acc_val_avg)
            plt.plot(k_grid, accuracies_val, 'ro-', label = "validation set")
            plt.plot(k_grid, accuracies_train, 'bo-', label = "training set")
            plt.xlabel("k")
            plt.ylabel("accuracy (%)")
            #plt.title("Finding the Optimal k: Performance Analysis of kNN (classification)")
            plt.legend()
            plt.show()
        elif plotKnnCenter:
            plt.figure()
            K = 5
            k_grid = [1, 5, 10, 20, 30, 40, 60, 80]
            mse_train = []
            mse_val   = []
            method_obj = KNN(0, "regression")
            for k in k_grid:
                method_obj.k = k
                mse_train_avg, mse_val_avg = KFold_cross_validation(xtrain, ctrain, K, method_obj)
                mse_train.append(mse_train_avg)
                mse_val.append(mse_val_avg)
            #plt.title(f"Finding the Optimal k: Performance Analysis of kNN (regression)")
            plt.plot(k_grid, mse_val, 'ro-', label = "validation set")
            plt.plot(k_grid, mse_train, 'bo-', label = "training set")
            plt.xlabel("k")
            plt.ylabel("mse")
            plt.legend()
            plt.show()



if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="center_locating", type=str, help="center_locating / breed_identifying")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / knn / linear_regression/ logistic_regression / nn (MS2)")
    parser.add_argument('--data_path', default="data", type=str, help="path to your dataset")
    parser.add_argument('--data_type', default="features", type=str, help="features/original(MS2)")
    parser.add_argument('--lmda', type=float, default=10, help="lambda of linear/ridge regression")
    parser.add_argument('--K', type=int, default=1, help="number of neighboring datapoints used for knn")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--KFold_plot', action="store_true", help="if true, use KFold cross validation and plot accuracy curves")
 
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--nn_type', default="cnn", help="which network to use, can be 'Transformer' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
