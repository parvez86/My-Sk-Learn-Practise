from sklearn.linear_model import Lasso
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.pipeline import make_pipeline
import numpy as np

data = datasets.load_boston()

# splitting into features and datasets
features = data.data
labels = data.target
# checking the loaded datasets
print(features.shape, labels.shape)

# splitting into train test datasets
train_set,  test_set, train_labels,test_labels = train_test_split(features, labels, train_size=0.8)

# checking the splitting
# print(train_set.shape)
# print(test_set.shape)
# print(train_labels.shape)
# print(test_labels.shape)

# Building Lasso Model
# model_ridge = Lasso(alpha = 0.1)

# using pipeline
model_lasso = make_pipeline(preprocessing.StandardScaler(), Lasso(alpha=0.1))

model_lasso.fit(train_set, train_labels)

print('Accuracy score:', model_lasso.score(test_set,test_labels))
# print('Model coefficients: ', model_ridge.classes_)
# print('Model intercept: ', model_ridge.intercept_)

# Using cross-validation

cv = ShuffleSplit(n_splits=1, train_size=0.7,)
print('Checking accuracy score(after CV):', cross_val_score(model_lasso, features, labels, cv = cv))

def get_cv(cv, data):
    train_indx = list()
    tst_indx = list()
    for train_index, test_index in cv.split(features):
        tr_indx = train_index
        tst_indx =test_index
    return tr_indx,tst_indx
train_index, test_index = get_cv(cv, features)
# train_index, test_index = cv.split(features)
model_lasso.fit(features[train_index], labels[train_index])
print('Accuracy score:', model_lasso.score(test_set,test_labels))
# print('Model coefficients: ', model_ridge.coef_)
# print('Model intercept: ', model_ridge.intercept_)