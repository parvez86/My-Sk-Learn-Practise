from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import load_boston
from sklearn.pipeline import make_pipeline

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
model_gb = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(), GradientBoostingRegressor(n_estimators=100))
model_gb.fit(train_set, train_labels)

# Observing the result
predictions = model_gb.predict(test_set)

print('Accuracy(GB Regressor): ', model_gb.score(test_set, test_labels))
print('MSE: ', mean_squared_error(test_labels, predictions))
print('Cross-Val accuracy: ', cross_val_score(model_gb,features_data, targets_data, cv=10))
print('Train score: ',model_gb['gradientboostingregressor'].train_score_)
print('Loss function: ',model_gb['gradientboostingregressor'].loss_)
print('Model important features: ', len(model_gb['gradientboostingregressor'].feature_importances_))
print('Model estimator: ', len(model_gb['gradientboostingregressor'].estimators_))
print('Model # features ', model_gb['gradientboostingregressor'].n_features_)
print('Model max features: ', model_gb['gradientboostingregressor'].max_features)