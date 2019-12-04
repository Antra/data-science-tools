import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Read the dataset
df = pd.read_csv('data/Weather.csv')
print(df.describe())


# do a quick plot and see if it's possible to eyeball anything
df.plot(x='MinTemp', y='MaxTemp', style='o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()


# and if we check the average temperature - we see that it's ~25 and 35
plt.figure(figsize=(15, 10))
plt.tight_layout()
seabornInstance.distplot(df['MaxTemp'])
plt.show()


# if we want to predict MaxTemp based on the recorded MinTemp, they need to be split into features (X) and labels (y)
# for a dataset with many columns: X = dataset[['fixed acidity', 'volatile acidity', 'citric acid']].values
X = df['MinTemp'].values.reshape(-1, 1)
y = df['MaxTemp'].values.reshape(-1, 1)

# and then split the set into training/test set - 80%:20% split in this case
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


# and then create the regression model and fit it to the data
regressor = LinearRegression()
regressor.fit(X_train, y_train)  # training the algorithm

# To retrieve the intercept:
print(regressor.intercept_)
# For retrieving the slope:
# for multi-columns: coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print(regressor.coef_)
# result is approx. 10.66185201 and 0.92033997 respectively; so for every 1 unit change in MinTemp the change in MaxTemp is about 0.92 (92% of a unit)


# now that the model is fitted, we can make predictions and score the model
y_pred = regressor.predict(X_test)
# for multi-column: df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df1)
# And plot the results -- as the results are huge, it may be better to just display e.g. head(25)
df2 = df1.head(25)
df2.plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# although the model is not very precise, the predicted percentages are close to the actuals - lets plot it with the test data (and actuals)
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


# The final step is to evaluate the performance of the model, commonly done as either
# 1) Mean Absolute Error (MAE) - the mean of the absolute value of the error
# 2) Mean Square Error (MSE) - the mean of the squared errors
# 3) Root Mean Square Error (RMSE) - the root of the mean of the squared errors
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(
    metrics.mean_squared_error(y_test, y_pred)))


# the value of root mean squared error is 4.19, which is more than 10% of the mean value of the percentages of all the temperature i.e. 22.41.
# This means that our algorithm was not very accurate but can still make reasonably good predictions.
