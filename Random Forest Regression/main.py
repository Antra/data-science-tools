# Random Forest Regressor with high accuracy, see: https://www.kaggle.com/nsrose7224/random-forest-regressor-accuracy-0-91

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os  # access os commands, for checking folders

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from subprocess import check_output
from datetime import time


print(os.listdir('data'))  # do we have files in the data folder?


# function to convert to seconds
def time_to_seconds(time):
    return time.hour * 3600 + time.minute * 60 + time.second


# define a list of typical missing values -- a dict can carry column specific NaN values
missing_values = ["n/a", "na", "--", "?"]
# read the data file - and encode missing values as np.nan
df = pd.read_csv("./data/data.csv", na_values=missing_values)
# Drop columns
df = df.drop("date", axis=1)
# Index(['number_people', 'timestamp', 'day_of_week', 'is_weekend', 'is_holiday', 'temperature', 'is_start_of_semester',
# 'is_during_semester', 'month', 'hour'], dtype = 'object')
print(df.columns)

# Numpy uses NaN, Pandas can find them wither either .isnull() or .isna()
print("Total number of missing values")
print(df.isnull().sum())  # We could also have used df.isna().sum()
# Good, no missing values.

# center timestamp
noon = time_to_seconds(time(12, 0, 0))
df.timestamp = df.timestamp.apply(lambda t: abs(noon - t))
# one hot encode categorical columns
columns = ["day_of_week", "month", "hour"]
df = pd.get_dummies(df, columns=columns)
print(df.head(10))  # 10 x 50 columns
print(df.shape)     # 62184, 50


# Extract the training and test data
data = df.values
X = data[:, 1:]  # all rows, no label
y = data[:, 0]  # all rows, label only
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


# Scale the data to be between -1 and 1
scaler = StandardScaler()
# Fit the scaler to the training set
scaler.fit(X_train)
# then apply it to both the training set and the test set
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Establish model
model = RandomForestRegressor(n_jobs=-1)


# Try different numbers of n_estimators - this will take 4-5 minutes
estimators = np.arange(10, 200, 10)  # range from 10-200 with 10 steps
scores = []
for n in estimators:
    model.set_params(n_estimators=n)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)
plt.show()  # what does the growth look like? When do we get the most gain and when does it level off?

print(scores)

# from Kaggle
#[0.906911897099843, 0.912672044577893, 0.9140890108660101, 0.9144505056433614, 0.914262282328547, 0.9139433343769434, 0.9163186431820812, 0.9166086725378967, 0.9160518483934068, 0.9167618968316904, 0.9165995923831073, 0.9172078680613065, 0.9164884688957832, 0.9161736086985683, 0.9165563453771916, 0.916971869620248, 0.9164909684909622, 0.9169219783868117, 0.9165771167149543]

# same thing but without dummy variable encoding -- slightly worse, so even with RF Regression it's a good idea to encode variables to dummies (even if it is only marginal difference)
#[0.9035434557721174, 0.911559395258506, 0.9116842411645403, 0.9137533295545505, 0.9145245791031484, 0.9140046860090325, 0.9147700440114759, 0.9156499566036366, 0.9156030346267331, 0.914918468907167, 0.914751144603725, 0.91524986942177, 0.9155485461437795, 0.9156745152151314, 0.9151416967009058, 0.9150332956454269, 0.9153819969209948, 0.9152587078577952, 0.9156019956278598]

# from Kaggle but without scaling -- difference is negligible, but it can make the model very vulnerable to high-value features
#[0.9073091037173004, 0.9117271973020955, 0.9134276558222074, 0.9149089218418027, 0.9146367394753295, 0.9167258249950438, 0.9151800535573515, 0.9162655888021767, 0.9165045474833977, 0.91607739466341, 0.9161604756415774, 0.9160704574077145, 0.9169646947491383, 0.9171442074438759, 0.9172018808756496, 0.917392225747372, 0.9168447733468765, 0.9170469690592035, 0.9170299413806312]
