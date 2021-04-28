from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import load_boston
from sklearn.pipeline import make_pipeline
from numpy import reshape,array
# load the datasets

data = load_boston()

features_name = data.feature_names
features_data = data.data

print('data size: ', features_data.shape)

targets_name = 'price'
targets_data = data.target

print('Features: ', features_name)
print('Targets: ', targets_name)

# Splitting into train-test datasets
train_set, test_set, train_labels, test_labels = train_test_split(features_data, targets_data, train_size=0.8)

print('Train data size: ', train_set.shape)
print('Test data size: ', test_set.shape)

# Building the model
model_rf = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(), RandomForestRegressor(max_depth=10, n_estimators=100))
model_rf.fit(train_set, train_labels)

# Observing the result
predictions = model_rf.predict(test_set)

print('Accuracy(RF Regressor): ', model_rf.score(test_set, test_labels))
print('MSE: ', mean_squared_error(test_labels, predictions))
print('Cross-Val accuracy: ', cross_val_score(model_rf,features_data, targets_data, cv=10))
print('Model important features: ', len(model_rf['randomforestregressor'].feature_importances_))
print('Model base estimators: ', model_rf['randomforestregressor'].base_estimator_)
# print('Model estimator: ', model_rf['randomforestregressor'].estimators_)
print('Model # features: ', model_rf['randomforestregressor'].n_features_)
print('Model # outputs: ', model_rf['randomforestregressor'].n_outputs_)