######################################################
# Diabetes Prediction with Logistic Regression
######################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation

# ############## Business Problem ###########
# Can you guess the probabilities human survival
# according to given the characteristics?
#############################################

# ############## Dataset Story ##############
# The dataset contains information about the people involved in the Titanic shipwreck.
# It consists of 768 observations and 12 variables.
# The target variable is specified as "Survived";
# 1 --> person to survive
# 0 --> indicates the death of the person.
#############################################

############################################
# Data Preprocessing
############################################

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate
import plotly.graph_objects as go
import plotly.offline as py
from helpers.data_prep import *
from helpers.eda import *

############################################
# Let's get the dataset
############################################

df = pd.read_csv("hafta_7/diabetes.csv")
df.head()

###########################################
# Analysis
############################################

# Summary statistics of all numeric variables:
# When I examine the dataset, we see that there is
# a difference between the mean and median values of the insulin variable.
# This difference is also supported by the standard deviation.
#  There is no missing value in dataset.
df.describe().T
check_df(df)
msno.bar(df)

# Frequencies are visually
sns.countplot(x="Outcome", data=df)
plt.show()

# Frequencies are visually
df["Insulin"].hist()

# Class ratios of the target variable:
100 * df["Outcome"].value_counts() / len(df)

# Checking how many samples we have for non-diabetics and for diabetics;
# We have more non-diabetic samples than diabetics.
# Also, we only have 786 samples total which is very less.
# Predictions would be more reliable if we had more data.

def pie_plot(cnt, colors, text):
    labels = list(cnt.index)
    values = list(cnt.values)

    trace = go.Pie(labels=labels,
                   values=values,
                   hoverinfo='value+percent',
                   title=text,
                   textinfo='label',
                   hole=.4,
                   textposition='inside',
                   marker=dict(colors=colors,
                               ),
                   )
    return trace

results = df["Outcome"].value_counts()
trace = pie_plot(results, ['#09efef', '#b70333'], "frequencies of the target")
py.iplot([trace], filename='results')


# We'll check mean values of the dependent variables for diabetics and non-diabetics
# From here, we can make deductions such as:
# Women with more pregnancies tend to be more diabetic.
# Non-diabetics tend to have lower glucose levels.
# Non-diabetics tend to have lower blood pressure.
# Diabetics have higher insulin levels.
# Diabetics weigh more than non-diabetics.
# Diabetics are older than non-diabetics on an average.
df.groupby('Outcome').mean()

###########################################
# Feature Engineering
###########################################
df.columns = [col.upper() for col in df.columns]

# About AGE
df.loc[(df['AGE'] < 13), 'NEW_AGE_CAT'] = 'CHILD'
df.loc[(df['AGE'] >= 13) & (df['AGE'] < 19), 'NEW_AGE_CAT'] = 'ADOLESCENCE'
df.loc[(df['AGE'] >= 19) & (df['AGE'] < 31), 'NEW_AGE_CAT'] = 'ADULT'
df.loc[(df['AGE'] >= 31) & (df['AGE'] < 60), 'NEW_AGE_CAT'] = 'SENIOR_ADULT'
df.loc[(df['AGE'] >= 60), 'NEW_AGE_CAT'] = 'OLD'

# About BMI
df.loc[(df['BMI'] < 18.5), 'NEW_BMI_CAT'] = 'UNDER_WEIGHT'
df.loc[(df['BMI'] >= 18.5) & (df['BMI'] < 24.9), 'NEW_BMI_CAT'] = 'NORMAL_WEIGHT'
df.loc[(df['BMI'] >= 24.9) & (df['BMI'] < 29.9), 'NEW_BMI_CAT'] = 'OVER_WEIGHT'
df.loc[(df['BMI'] >= 29.9), 'NEW_BMI_CAT'] = 'OBESE'

# About PREGNANCIES
df.loc[(df['PREGNANCIES'] < 3), 'NEW_PREG_CAT'] = 'LESS'
df.loc[(df['PREGNANCIES'] >= 3) & (df['PREGNANCIES'] < 6), 'NEW_PREG_CAT'] = 'NORMAL'
df.loc[(df['PREGNANCIES'] >= 6), 'NEW_PREG_CAT'] = 'HIGH'

# About BLOOD PRESSURE
df.loc[(df['BLOODPRESSURE'] < 80), 'NEW_BLOOD_PRES'] = 'NORMAL'
df.loc[(df['BLOODPRESSURE'] >= 80) & (df['BLOODPRESSURE'] < 89), 'NEW_BLOOD_PRES'] = 'HIGH'
df.loc[(df['BLOODPRESSURE'] >= 89) & (df['BLOODPRESSURE'] < 120), 'NEW_BLOOD_PRES'] = 'SO_HIGH'
df.loc[(df['BLOODPRESSURE'] >= 120), 'NEW_BLOOD_PRES'] = 'CRITIC'

# About INSULIN
df.loc[(df['INSULIN'] < 100), 'NEW_INSULIN'] = 'NORMAL'
df.loc[(df['INSULIN'] >= 100) & (df['INSULIN'] < 126), 'NEW_INSULIN'] = 'PRE_DIABETES'
df.loc[(df['INSULIN'] >= 126), 'NEW_INSULIN'] = 'DIABETES'

# About GLUCOSE
df.loc[(df['GLUCOSE'] < 50), 'GLUCOSE_CAT'] = 'LESS'
df.loc[(df['GLUCOSE'] >= 50) & (df['GLUCOSE'] < 141), 'GLUCOSE_CAT'] = 'NORMAL'
df.loc[(df['GLUCOSE'] >= 141) & (df['GLUCOSE'] < 200), 'GLUCOSE_CAT'] = 'HIGH'
df.loc[(df['GLUCOSE'] >= 200), 'GLUCOSE_CAT'] = 'SO_HIGH'

###########################################
# Missing Value Analysis
###########################################
df.isnull().sum()

zero_cols = ["GLUCOSE", "BLOODPRESSURE", "SKINTHICKNESS", "INSULIN", "BMI"]

for col in zero_cols:
    df.loc[(df[col] == 0), col] = df[col].median()

df.isnull().any().sum()
###########################################
# Outlier Analysis
###########################################
col_name = df.columns

outlier_thresholds(df, col_name)
check_outlier(df, col_name)
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# As we can see below, we detected an outlier in the insulin value
# that we noticed in the describe function.
for col in num_cols:
    print(col, check_outlier(df, col))

# We suppressed it on threshold values.
for col in num_cols:
    replace_with_thresholds(df, col)

# We have eliminated the outlier situation.
replace_with_thresholds(df, "INSULIN")
for col in num_cols:
    print(col, check_outlier(df, col))

# Target analysis group by Outcome
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "OUTCOME", col)

# In the feature engineering section, we produced new variables
# thanks to the variables in dummies list below. For this reason,
# we extract the variables from our dataset.
dummies = ['PREGNANCIES','BMI', 'AGE']
dummy_data = pd.get_dummies(df[dummies])
df = pd.concat([df, dummy_data], axis = 1)
df.drop(dummies, axis=1, inplace=True)
df.head()

###########################################
# Corr Between Variables
###########################################
# We can make observations such as, as glucose increases,
# the likelihood of having diabetes increases.
df.corr()
sns.heatmap(df.corr(), annot=True)

# Age vs Glucose
# No real trend found. But, interestingly 22 year olds have the highest glucose
# in their blood and 68 year olds have the least.
x = []

for age in df.Age:
    x.append(age)
y = df.Glucose
plt.figure(figsize=(20, 10))
plt.bar(x, y)
plt.xlabel("AGE", size=10)
plt.ylabel("Glucose")
plt.xticks(x)
plt.grid()
plt.title("Relationship between Age and Glucose levels")
plt.show()

###########################################
# Feature Analysis
###########################################

df["BLOODPRESSURE"].hist(bins=20)
plt.xlabel("BLOODPRESSURE")
plt.show()


def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show()

cols = [col for col in df.columns if "OUTCOME" not in col]

for col in cols:
    plot_numerical_col(df, col)

###########################################
# One-Hot Encoding
###########################################
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)
cat_cols, num_cols, cat_but_car = grab_col_names(df)

###########################################
# Rare Encoding
###########################################
rare_analyser(df, "OUTCOME", cat_cols)
rare_encoder(df, 0.01, cat_cols)

df.head()
###########################################
# Data Standardization
###########################################
# All our columns have variable ranges. This could cause some problems
# for our ML model to make prediction. So, we should standardize the data,
# bring all of it to a similar range so it's easy for our model to understand the data.

for col in num_cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

###########################################
# Splitting Dataset
###########################################

# We need all the variables (columns) as independent variables so
# we are just dropping the target column to make things easier.
X = df.drop(['OUTCOME'], axis=1)
y = df['OUTCOME']  # Target.

###########################################
# Model Validation: Holdout
###########################################
# Then we split the data into training and testing data
# 80% data will be used for training the model and rest 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 2)

# As we can see, 614 rows are used for testing out of 768 which is about 79.9% of the data.
X.shape
X_train.shape

log = LogisticRegression()
log.fit(X_train, y_train)

y_pred = log.predict(X_test)

# Classification report
print(classification_report(y_test, y_pred))
###################################################
#    precision   recall   f1-score   support
# 0     0.84      0.87      0.86       109
# 1     0.66      0.60      0.63        45
###################################################

# ROC Curve
plot_roc_curve(log, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

######################################################
# Model Validation: 10-Fold Cross Validation
######################################################
# We apply cross validation over the fit above.
cv_results = cross_validate(log,
                            X, y,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])


cv_results['test_accuracy'].mean()
# Accuracy: 0.7721

cv_results['test_precision'].mean()
# Precision: 0.7218

cv_results['test_recall'].mean()
# Recall: 0.5709

cv_results['test_f1'].mean()
# F1-score: 0.6361

cv_results['test_roc_auc'].mean()
# AUC: 0.8451

######################################################
# Prediction for A New Observation
#####################################################
X.columns

random_user = X.sample(1, random_state=44)
log.predict(random_user)
