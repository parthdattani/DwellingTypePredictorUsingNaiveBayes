# Naive Bayes Model for Predicting DwellingType

This repository contains the code for building a Naive Bayes model to predict the DwellingType of a property based on other variables in the dataset.

## Installation

To install the required packages, run the following command in your terminal:

```bash
!pip install pandas scikit-learn
```
The code for building the Naive Bayes model is provided in the dwellingTypePrediction.py file. The code can be summarized as follows:

Import the required libraries, including pandas and scikit-learn.
Load the DwellingType.csv dataset into a pandas DataFrame.
Split the dataset into training and testing sets.
Train a Naive Bayes model using the training set.
Predict the DwellingType for the records in the testing set.
Evaluate the model's accuracy using a confusion matrix and accuracy score.

## Running the Code

To run the code, save the dwellingTypePrediction.py file in your working directory and execute the following command in your terminal:

```bash
python dwellingTypePrediction.py
```

The code will output the following:

1. The DwellingType dataset
2. The structure of the DwellingType dataset
3. A summary of the DwellingType dataset
4. The Naive Bayes model's prediction probabilities
5. The Naive Bayes model's predictions
6. The Naive Bayes model's confusion matrix
7. The Naive Bayes model's accuracy score

