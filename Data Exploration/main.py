import pandas as pd
import numpy as np

# define types of missing values -- a dict can carry column specific NaN values
missing_values = ['n/a', '-', '?']
# read the data file - and encode missing values as np.nan
df = pd.read_csv("./data/data.csv", na_values=missing_values)
# fix missing with fill-up as 0
df.fillna(0, inplace=True)
print(df.head())
# Change one columns data type
df['NUM_BATH'] = df['NUM_BATH'].astype(np.int32)
# which data types do we have?
print(df.dtypes)
# make some columns list - either with the specific types or use 'np.number'
num_dtypes = ['int64', 'float64', 'int32', 'float32']
num_cols = df.select_dtypes([np.number]).columns.tolist()

print("The following columns are numeric:", num_cols)
