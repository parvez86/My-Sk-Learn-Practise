from sklearn.ensemble import GradientBoostingClassifier
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
model_gb = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(), GradientBoostingClassifier(n_estimators=100))
model_gb.fit(train_set, train_labels)

# Observing the result
predictions = model_gb.predict(test_set)

print('Accuracy(GB Classifier): ', model_gb.score(test_set, test_labels))
print('Accuracy: ', accuracy_score(test_labels, predictions))
print('Cross-Val accuracy: ', cross_val_score(model_gb,features_data, targets_data, cv=10))
print('Train score: ',model_gb['gradientboostingclassifier'].train_score_)
print('Loss function: ',model_gb['gradientboostingclassifier'].loss_)
print('Model important features: ', len(model_gb['gradientboostingclassifier'].feature_importances_))
print('Model # estimators: ', model_gb['gradientboostingclassifier'].n_estimators_)
print('Model estimator: ', len(model_gb['gradientboostingclassifier'].estimators_))
print('Model # features ', model_gb['gradientboostingclassifier'].n_features_)
print('Model max features: ', model_gb['gradientboostingclassifier'].max_features)
print('Confusion Matrix: \n', confusion_matrix(predictions, test_labels))
print('Classification report: \n', classification_report(predictions, test_labels))