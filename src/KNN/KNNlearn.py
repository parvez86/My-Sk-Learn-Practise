import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car.data')

# splitting features
x = data[['buying', 'maint', 'safety']].values
y = pd.DataFrame()
# y = data[['class']]
# print(x, y)

# converting X to the data
Le = LabelEncoder()
for i in range(len(x[0])):
    x[:, i] = Le.fit_transform(x[:, i])
print(x)

# set labels(y) in the class data
label_mapping = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3
}
target_names = {
    0: 'unacc',
    1: 'acc',
    2: 'good',
    3: 'vgood'
}
y['class'] = data['class'].map(label_mapping)
y = np.array(y)
# print(y)

# Building KNN model

knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')
knn2 = neighbors.KNeighborsClassifier(n_neighbors=25, weights='distance')

# splitting train-test data
train_set, test_set, train_labels, test_labels = train_test_split(x, y, train_size=0.8)

# training the model by fit()
knn.fit(train_set, train_labels.ravel())
knn2.fit(train_set, train_labels.ravel())

# make predictions
predictions = knn.predict(test_set)
predictions2 = knn2.predict(test_set)

# evaluate model accuracy
accuracy = metrics.accuracy_score(predictions, test_labels)
accuracy2 = metrics.accuracy_score(predictions2, test_labels)
print('Accuray(by uniform metrics):', accuracy)
print('Accuray(by distance metrics):', accuracy2)

sample = [[5, 5, 3], [2, 4, 3]]
preds = knn.predict(sample)
preds2 = knn2.predict(sample)
print('On sample data: ')
pred_species = [target_names[p] for p in preds]
print("Predictions:", pred_species)
pred_species2 = [target_names[p] for p in preds2]
print("Predictions:", pred_species2)
