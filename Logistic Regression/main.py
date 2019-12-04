import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import seaborn as sns


# Read the dataset
df = pd.read_csv('data/bank-additional-full.csv', sep=';')
print(df.describe())

# Split into predictor and response dataframes.
X_df = df.drop('y', axis=1)
y = df['y']

# Map response variable to integers 0,1 - instead of yes/no
y = pd.Series(np.where(y.values == 'yes', 1, 0), y.index)


# Separating with continuous and categorical variables.
X_cont = ['age', 'campaign', 'pdays', 'previous',
          'emp.var.rate', 'cons.price.idx', 'euribor3m', 'nr.employed']
X_cat = ['job', 'marital', 'education', 'default', 'housing',
         'loan', 'contact', 'month', 'day_of_week', 'poutcome']
cont_df = X_df[X_cont]
cat_df = X_df[X_cat]


# To run logistic regression on the data, we need to convert all the non-numeric features to numeric ones.
# Two popular ways are: Label encoding and one hot encoding
# Let's fit one model only with dummy variables and one with only label encoded variables

# Start with Dummy Variables
X_df = cont_df.join(pd.get_dummies(cat_df))


# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=10)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Initial logistic regression - with dummy variables
clf = LogisticRegression(C=0.001)
model_base = clf.fit(X_train, y_train)

# and calculate metrics
y_pred = model_base.predict(X_test)
model_base.score(X_test, y_test)
print("Model accuracy is", model_base.score(X_test, y_test))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))

# Print ROC curve
probs = model_base.predict_proba(X_test)
preds = probs[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
logit_roc_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, label='Logistic Regression Base (area = %0.2f)' %
         logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
# plt.savefig('Log_ROC')
plt.show()


# Using Label Encoding
mappings = []
label_encoder = LabelEncoder()

label_df = df.drop('y', axis=1)
for i, col in enumerate(label_df):
    if label_df[col].dtype == 'object':
        label_df[col] = label_encoder.fit_transform(
            np.array(label_df[col].astype(str)).reshape((-1,)))
        mappings.append(
            dict(zip(label_encoder.classes_, range(1, len(label_encoder.classes_)+1))))

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    label_df, y, test_size=0.2, random_state=10)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Initial logistic regression - with label encoding
clf = LogisticRegression()
model_label = clf.fit(X_train, y_train)

# and calculate metrics
y_pred = model_label.predict(X_test)
model_label.score(X_test, y_test)
print("Model accuracy is", model_label.score(X_test, y_test))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))


# Print ROC curve
probs = model_label.predict_proba(X_test)
preds = probs[:, 1]
labelfpr, labeltpr, labelthreshold = metrics.roc_curve(y_test, preds)
label_roc_auc = metrics.auc(labelfpr, labeltpr)

plt.figure()
plt.plot(labelfpr, labeltpr,
         label='Logistic Regression with LabelEncoder(area = %0.2f)' % label_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

print(mappings)


# What if we use a mixture?
educ_order = ['unknown', 'illiterate', 'basic.4y', 'basic.6y',
              'basic.9y', 'high.school', 'professional.course', 'university.degree']
month_order = ['mar', 'apr', 'may', 'jun',
               'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
day_order = ['mon', 'tue', 'wed', 'thu', 'fri']


# using cat.codes for order, one hot for high cardinality and weak case of cardinality.
def ordered_labels(df, col, order):
    df[col] = df[col].astype('category')
    df[col] = df[col].cat.reorder_categories(order, ordered=True)
    df[col] = df[col].cat.codes.astype(int)


mappings2 = []
label_df_2 = df.drop('y', axis=1)
# Use dummy variables for occupation
label_df_2 = pd.concat([label_df_2, pd.get_dummies(
    label_df_2['job'])], axis=1).drop('job', axis=1)

# Use ordered cat.codes for days, months, and education
ordered_labels(label_df_2, 'education', educ_order)
ordered_labels(label_df_2, 'month', month_order)
ordered_labels(label_df_2, 'day_of_week', day_order)

# Same label encoding for rest since low cardinality
for i, col in enumerate(label_df_2):
    if label_df_2[col].dtype == 'object':
        label_df_2[col] = label_encoder.fit_transform(
            np.array(label_df_2[col].astype(str)).reshape((-1,)))
        mappings2.append(
            dict(zip(label_encoder.classes_, range(1, len(label_encoder.classes_)+1))))


#


# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    label_df_2, y, test_size=0.2, random_state=10)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Initial logistic regression - with dummy variables
clf = LogisticRegression()
model_mix = clf.fit(X_train, y_train)

# and calculate metrics
y_pred = model_mix.predict(X_test)
model_mix.score(X_test, y_test)
print("Model accuracy is", model_mix.score(X_test, y_test))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))

# Print ROC curve
classes = model_mix.predict(X_test)
probs = model_mix.predict_proba(X_test)
preds = probs[:, 1]
mixfpr, mixtpr, mixthreshold = metrics.roc_curve(y_test, preds)
mix_roc_auc = metrics.auc(mixfpr, mixtpr)

plt.figure()
plt.plot(mixfpr, mixtpr,
         label='Logistic Regression Mixed Labels (area = %0.2f)' % mix_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# Model scoring
print(metrics.classification_report(classes, y_test))


# Feature importance rankings
weights = zip(label_df_2.columns, model_mix.coef_[0])
ranked_weights = sorted(weights, key=lambda x: x[1], reverse=True)
# Top 10 positive feature importance
print(ranked_weights[:10])
# Top 10 negative feature importance
ranked_weights[-10:]


# Absolute value feature importance
abs_weights = zip(label_df_2.columns, model_mix.coef_[0])
abs_ranked_weights = sorted(abs_weights, key=lambda x: abs(x[1]), reverse=True)
abs_ranked_weights[:10]
# Absolute value feature importance plot
labels, weights = zip(*abs_ranked_weights[:10])
sns.barplot(x=pd.Series(labels), y=pd.Series(
    weights), color='aquamarine', alpha=0.8)
plt.xticks(rotation=40, ha='right')
plt.show()
