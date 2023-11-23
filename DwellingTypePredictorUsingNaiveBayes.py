# Install required packages
# !pip install pandas scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Set the working directory to Lab08 folder
# Replace the path with your actual path
working_directory = "C:/Users/Apple Kaur/Desktop/FALL1 2022/MIS545 Data Mining/Rlab08Naives"
dwellingType_path = "DwellingType.csv"
dwellingType_full_path = f"{working_directory}/{dwellingType_path}"

# Reading DwellingType.csv into a pandas DataFrame
dwellingType = pd.read_csv(dwellingType_full_path)

# Displaying dwellingType in the console
print(dwellingType)

# Displaying the structure of dwellingType in the console
print(dwellingType.info())

# Displaying the summary of dwellingType in the console
print(dwellingType.describe())

# Randomly splitting the dataset into dwellingTypeTraining (75% of records) 
# and dwellingTypeTesting (25% of records) using 154 as the random seed
dwellingTypeTraining, dwellingTypeTesting = train_test_split(dwellingType, test_size=0.25, random_state=154)

# Generating the Naive Bayes model to predict DwellingType based on the 
# other variables in the dataset
features = dwellingType.columns[:-1]
target = 'DwellingType'
dwellingTypeModel = GaussianNB()
dwellingTypeModel.fit(dwellingTypeTraining[features], dwellingTypeTraining[target])

# Building probabilities for each record in the testing dataset and
# storing them in dwellingTypeProbability
dwellingTypeProbability = dwellingTypeModel.predict_proba(dwellingTypeTesting[features])

# Displaying dwellingTypeProbability on the console
print(dwellingTypeProbability)

# Predicting classes for each record in the testing dataset and storing 
# them in dwellingTypePrediction
dwellingTypePrediction = dwellingTypeModel.predict(dwellingTypeTesting[features])

# Displaying dwellingTypePrediction on the console
print(dwellingTypePrediction)

# Evaluating the model by forming a confusion matrix
dwellingTypeconfusionMatrix = confusion_matrix(dwellingTypeTesting[target], dwellingTypePrediction)

# Displaying the confusion matrix on the console
print(dwellingTypeconfusionMatrix)

# Calculating the model predictive accuracy and store it into a 
# variable called predictiveAccuracy
predictiveAccuracy = accuracy_score(dwellingTypeTesting[target], dwellingTypePrediction)

# Displaying the predictive accuracy on the console
print(predictiveAccuracy)
