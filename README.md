# Diabetes Prediction with Logistic Regression


1. Exploratory Data Analysis
2. Data Preprocessing
3. Model & Prediction
4. Model Evaluation
5. Model Validation: Holdout
6. Model Validation: 10-Fold Cross Validation
7. Prediction for A New Observation

# Business Problem 
Characteristics of people with diabetes will be able to predict whether they have a patient or not it is desirable to develop a machine learning model.


# Dataset Story

The data set is part of a large data set maintained at the National Institutes of Diabetes-dIgestive-Kidney Diseases in the United States. this data used for a diabetes study conducted on Pima Indian women aged 21 years and older living in the city of Phoenix, which is their city. The data consists of 768 observations and 8 numerical independent variables. The target variable is specified as "output";

1 diabetes test result is positive, 0 indicates that it is negative.

# Variables

- Pregnancies: Number of pregnancies
- Glucose: 2 Hours plasma glucose concentration in the oral glucose tolerance test
- Blood Pressure: mm Hg
- SkinThickness:
- Insulin: 2 Hours serum insulin (mu U/ml)
- DiabetesPedigreeFunction
- Age: years
- Outcome: Having diabete (1) or not (0)


In this study, the diabetes data set was reviewed and it was tried to predict whether a person has diabetes with a Logistic Regression model. Firstly, the dependent variable "outcome" was reviewed in the study. In the last step, new variables were produced and the success of the model was tried to be increased. The accuracy rate and F1 score of the established model were determined as 0.63 and the AUC value was determined as 0.84. Finally, it was estimated by the established model whether a randomly selected person has diabetes or not.
