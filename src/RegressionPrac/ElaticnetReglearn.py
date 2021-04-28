from sklearn.linear_model import ElasticNet
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.metrics import accuracy_score
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

# Building Elasticnet Model
model_elasticnet = ElasticNet(alpha=0.1)

# using pipeline
# model_elasticnet = make_pipeline(preprocessing.StandardScaler(), ElasticNet(alpha=0.1))

model_elasticnet.fit(train_set,train_labels)

prdictions = model_elasticnet.predict(test_set)
print('Accuracy score:', model_elasticnet.score(test_set, test_labels))
print('Model coefficients: ', model_elasticnet.coef_)
print('Model sparse coefficients; ', model_elasticnet.sparse_coef_)
print('Model intercept: ', model_elasticnet.intercept_)

# Using cross-validation
cv = ShuffleSplit(n_splits=2, train_size=0.7,)
print('Checking accuracy score(after CV):', cross_val_score(model_elasticnet, features, labels, cv = cv))


def get_cv(cv, data):
    train_indx = list()
    tst_indx = list()
    for train_index, test_index in cv.split(features):
        tr_indx = train_index
        tst_indx =test_index
    return tr_indx,tst_indx


train_index, test_index = get_cv(cv, features)
# train_index, test_index = cv.split(features)
model_elasticnet.fit(features[train_index], labels[train_index])
print('Accuracy score:', model_elasticnet.score(test_set,test_labels))
print('Model coefficients: ', model_elasticnet.coef_)
print('Model sparse coefficients; ', model_elasticnet.sparse_coef_)
print('Model intercept: ', model_elasticnet.intercept_)