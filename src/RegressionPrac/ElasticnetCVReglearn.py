from sklearn.linear_model import ElasticNetCV
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

# Building ElasticnetCV Model
model_elasticnetcv = ElasticNetCV(cv=5)

# using pipeline
# model_elasticnet = make_pipeline(preprocessing.StandardScaler(), ElasticNetCV(alpha=0.1,cv=3))

model_elasticnetcv.fit(train_set,train_labels)

# predictions = model_elasticnet.predict(test_set)
print('Accuracy score:', model_elasticnetcv.score(test_set, test_labels))
print('Model coefficients: ', model_elasticnetcv.coef_)
print('Model alpha:  ', model_elasticnetcv.alpha_)
print('Model intercept: ', model_elasticnetcv.intercept_)