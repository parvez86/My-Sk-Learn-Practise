from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline
from numpy import reshape,array
# load the datasets

data = load_breast_cancer()

features_name = data.feature_names
features_data = data.data

print('data size: ', features_data.shape)

targets_name = data.target_names
targets_data = data.target

print('Features: ', features_name)
print('Targets: ', targets_name)

# Splitting into train-test datasets
train_set, test_set, train_labels, test_labels = train_test_split(features_data, targets_data, train_size=0.8)

print('Train data size: ', train_set.shape)
print('Test data size: ', test_set.shape)

# Building the model
model_sgd = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3,))
model_sgd.fit(train_set, train_labels)

# Observing the result
predictions = model_sgd.predict(test_set)

print('Accuracy(SGD Classifier): ', model_sgd.score(test_set, test_labels))
print('Accuracy: ', accuracy_score(test_labels, predictions))
print('Model co-efficient: ', model_sgd['sgdclassifier'].coef_)
print('Model intercept: ', model_sgd['sgdclassifier'].intercept_)
print('Model intercept: ', model_sgd['sgdclassifier'].loss_function_)
# print('Model decission function: ', model_sgd['sgdclassifier'].decision_function(reshape(test_set,(569, 30))))
print('Sparsify: ', model_sgd['sgdclassifier'].sparsify())