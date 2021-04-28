from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

# loading the boston house pricing data
data = datasets.load_iris()

# splitting into features and datasets
features = data.data
labels = data.target
# checking the loaded datasets
print(features.shape, labels.shape)

# splitting into train test datasets
train_set,  test_set, train_labels,test_labels = train_test_split(features, labels, train_size=0.8)

# checking the splitting
print(train_set.shape)
print(test_set.shape)
print(train_labels.shape)
print(test_labels.shape)


# Building the model
model = LogisticRegression(random_state=0 ,solver='liblinear', multi_class= 'auto')
model.fit(train_set, train_labels)

# making predictions and getting accuracy
predictions = model.predict(test_set)

print('Linear Regression score: ',model.score(features, labels))
print('Model intercept: ', model.intercept_)
print('Model co_efficients: ', model.coef_)

# viewing the result
for i in range(len(predictions)):
    print('Predict label: ', data.target_names[predictions[i]], '\tActual label: ', data.target_names[test_labels[i]])
