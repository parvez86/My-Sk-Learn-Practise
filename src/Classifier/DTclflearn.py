from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_val_score
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
model_DTC = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(), DecisionTreeClassifier())
model_DTC.fit(train_set, train_labels)

#
predictions = model_DTC.predict(test_set)

print('Accuracy(DT Classifier): ', model_DTC.score(test_set, test_labels))
print('Accuracy: ', accuracy_score(test_labels, predictions))
print('Cross-Val accuracy: ', cross_val_score(model_DTC,features_data, targets_data, cv=10))
print('Model important features: ', len(model_DTC['decisiontreeclassifier'].feature_importances_))
print('Model # features: ', model_DTC['decisiontreeclassifier'].n_features_)
print('Model # outputs: ', model_DTC['decisiontreeclassifier'].n_outputs_)
print('Model tree: ', model_DTC['decisiontreeclassifier'].tree_)
