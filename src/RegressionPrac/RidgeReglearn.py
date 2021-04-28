from sklearn.linear_model import Ridge
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



# Building Ridge Model
# model_ridge = Ridge(alpha = 0.5)

# using pipeline
model_ridge = make_pipeline(preprocessing.PolynomialFeatures(degree=2), Ridge(alpha=0.1))
model_ridge.fit(train_set, train_labels)

print('Accuracy score:', model_ridge.score(test_set,test_labels))
# print('Model coefficients: ', model_ridge.classes_)
# print('Model intercept: ', model_ridge.intercept_)

# Using cross-validation

cv = ShuffleSplit(n_splits=2, train_size=0.7,)
print('Checking accuracy score(after CV):', cross_val_score(model_ridge, features, labels, cv = cv))

def get_cv(cv, data):
    train_indx = list()
    tst_indx = list()
    for train_index, test_index in cv.split(features):
        tr_indx = train_index
        tst_indx =test_index
    return tr_indx,tst_indx
train_index, test_index = get_cv(cv, features)
# train_index, test_index = cv.split(features)
model_ridge.fit(features[train_index], labels[train_index])
print('Accuracy score:', model_ridge.score(test_set,test_labels))
# print('Model coefficients: ', model_ridge.coef_)
# print('Model intercept: ', model_ridge.intercept_)