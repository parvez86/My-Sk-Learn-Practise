from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline

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
model_adb = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(), AdaBoostClassifier(n_estimators=100))
model_adb.fit(train_set, train_labels)

# Observing the result
predictions = model_adb.predict(test_set)

print('Accuracy(ADB Classifier): ', model_adb.score(test_set, test_labels))
print('MSE: ', accuracy_score(test_labels, predictions))
print('Cross-Val accuracy: ', cross_val_score(model_adb,features_data, targets_data, cv=10))
print('Model important features: ', len(model_adb['adaboostclassifier'].feature_importances_))
print('Model base estimators: ', model_adb['adaboostclassifier'].base_estimator_)
# print('Model estimator: ', model_adb['adaboostclassifier'].estimators_)
print('Model estimator weights: ', model_adb['adaboostclassifier'].estimator_weights_)
print('Model estimator errors: ', model_adb['adaboostclassifier'].estimator_errors_)
print('Confusion Matrix: ', confusion_matrix(predictions, test_labels))
print('Classification report: \n', classification_report(predictions, test_labels))