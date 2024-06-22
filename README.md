# Stanford-Dogs

## Machine Learning Project

**Authors**: Ahmed Tlili, Cyrine Akrout, Ahmed Zguir  
**Institution**: EPFL  
**Date**: April 20, 2024

## Introduction

This repository contains the implementation of three machine learning models: `LinearRegression`, `LogisticRegression`, and `KNN`. These models are utilized to train and predict on a given dataset of dog breeds.

The Stanford Dogs dataset is a popular dataset used in computer vision tasks. It contains images of 120 breeds of dogs from around the world, and it is commonly used for image classification and object localization tasks. 

In this project, we aim to achieve two tasks using the Stanford Dogs dataset:
1. **Center**: Locate the center of the dog in the image.
2. **Breed**: Determine the breed of the dog in the image.

For a detailed walkthrough, check the [report.pdf](report.pdf) file.

## Usage

To run the models with optimized hyperparameters, use the following commands:

1. **Linear Regression for Center Locating**
    ```bash
    python main.py --method linear_regression --lmda 0 --task center_locating --test
    ```

2. **Logistic Regression for Breed Identifying**
    ```bash
    python main.py --method logistic_regression --lr 1e-3 --max_iters 600 --task breed_identifying --test
    ```

3. **KNN for Center Locating**
    ```bash
    python main.py --method knn --task center_locating --K 30 --test
    ```

4. **KNN for Breed Identifying**
    ```bash
    python main.py --method knn --task breed_identifying --K 20 --test
    ```

## Results

The overall performance after optimization of hyperparameters and running the above commands is summarized in the table below:

| -        | Linear Reg       | Logistic Reg    | KNN             |
|----------|------------------|-----------------|-----------------|
| Breed    | -                | Accuracy: 86.2% | Accuracy: 86.2% |
| Center   | MSE: 0.005       | -               | MSE: 0.005      |


## Conclusion

This project demonstrates the effective implementation and evaluation of different machine learning algorithms for regression and classification tasks. The results indicate that careful tuning of hyperparameters is crucial for achieving optimal performance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
