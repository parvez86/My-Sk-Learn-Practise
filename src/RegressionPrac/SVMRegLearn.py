from sklearn.svm import SVR
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

model_reg = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), SVR(kernel='sigmoid',max_iter=1000, tol=1e-3))
model_reg.fit(train_set, train_labels)

# Getting model atributes
print('Accurracy score(SVM Reg): ', model_reg.score(test_set, test_labels))
print('Model support: ', model_reg['svr'].support_)
print('Model support vectors: ', model_reg['svr'].support_vectors_)
print('Model dual coefficients: ', model_reg['svr'].dual_coef_)
print('Model intercepts: ', model_reg['svr'].intercept_)
print('Model fit status: ', model_reg['svr'].fit_status_)