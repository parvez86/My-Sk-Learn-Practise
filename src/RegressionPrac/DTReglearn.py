from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.datasets import load_boston
from sklearn.pipeline import make_pipeline

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

# checking datasets
print('Train data size: ', train_set.shape)
print('Test data size: ', test_set.shape)

# Building the model
# model_sgd = SGDRegressor()

model_reg = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), DecisionTreeRegressor())
model_reg.fit(train_set, train_labels)

# Getting model atributes
print('Accurracy score(SGD Reg): ', model_reg.score(test_set, test_labels))
print('Model important features: ', len(model_reg['decisiontreeregressor'].feature_importances_))
print('Model # features: ', model_reg['decisiontreeregressor'].n_features_)
print('Model # outputs: ', model_reg['decisiontreeregressor'].n_outputs_)
print('Model tree: ', model_reg['decisiontreeregressor'].tree_)