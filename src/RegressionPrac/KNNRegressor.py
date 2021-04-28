from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.datasets import load_boston
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt

# collecting data
boston = load_boston()

# splitting into features & targets
features_name = boston.feature_names
features_val = boston.data

target_name = 'price'
target_val = boston.target

print('Features: ', features_name)
print('Targets: ', target_name)


# Splitting into train, test datasets
train_set, test_set, train_labels, test_labels = train_test_split(features_val, target_val, train_size=0.75)
print('train size: ', train_set.shape)
print('test size: ', test_set.shape)
print('train label size: ', train_labels.shape)
print('test label size: ', test_labels.shape)

model_reg = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), KNeighborsRegressor(n_neighbors=5))
model_reg.fit(train_set, train_labels)

prediction = model_reg.predict(test_set)

# Getting model atributes
print('Accurracy score(KNeighbors Regressor): ', model_reg.score(test_set, test_labels))
print('MSE: ', np.sqrt((model_reg.predict(test_set)-test_labels)**2).mean())
