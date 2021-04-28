from sklearn import datasets
from sklearn.model_selection import train_test_split
# import numpy as np

# loading datasets
iris = datasets.load_iris()

# splitting features & labels
features = iris.data
labels = iris.target

# feature & targets
print(iris.feature_names)
print(iris.target_names)

# checking splitting
print(features.shape)
print(labels.shape)

# splitting into train test datasets
train_set, test_set, train_labels, test_labels = train_test_split(features, labels, train_size=0.8)

print(train_set.shape)
print(test_set.shape)
print(train_labels.shape)
print(test_labels.shape)
