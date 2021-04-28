from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
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
model_rf = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(), RandomForestClassifier(n_estimators=50))
model_rf.fit(train_set, train_labels)

# Observing the result
predictions = model_rf.predict(test_set)

print('Accuracy(RF Classifier): ', model_rf.score(test_set, test_labels))
print('Accuracy: ', accuracy_score(test_labels, predictions))
print('Cross-Val accuracy: ', cross_val_score(model_rf,features_data, targets_data, cv=10))
print('Model important features: ', len(model_rf['randomforestclassifier'].feature_importances_))
print('Model base estimators: ', model_rf['randomforestclassifier'].base_estimator_)
print('Model estimator: ', model_rf['randomforestclassifier'].estimators_)
print('Model # features: ', model_rf['randomforestclassifier'].n_features_)
print('Model # outputs: ', model_rf['randomforestclassifier'].n_outputs_)
print('Model tree: ', model_rf['randomforestclassifier'].n_classes_)
print('Confusion matrix: ',confusion_matrix(predictions, test_labels))
print('Classification report: \n',classification_report(predictions, test_labels))