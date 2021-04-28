from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from sklearn import metrics
from time import time
import pandas as pd
import numpy as np

# loading data
data = load_breast_cancer()

# splitting into  features, targets

features_name = data.feature_names
features_data = data.data

targets_name = data.target_names
labels_data = data.target

# splitting into train, test data
train_set, test_set, train_labels, test_labels = train_test_split(features_data, labels_data, train_size=0.8)

# print(train_set.shape)
# print(test_set.shape)

# Building KMeans model
model = KMeans(n_clusters=len(targets_name), random_state=0)
model.fit(train_set)

predictions = model.predict(test_set)
accuracy = accuracy_score(test_labels, predictions)
print('KMeans accuracy: ',accuracy)
# print('predictions_labels: ', model.labels_)

# displaying results analysis
# print(np.count_nonzero(test_labels))
print('Confusion matrix: \n', pd.crosstab(predictions,test_labels, margins=True, margins_name= 'Total'))

def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels_data, estimator.labels_),
             metrics.completeness_score(labels_data, estimator.labels_),
             metrics.v_measure_score(labels_data, estimator.labels_),
             metrics.adjusted_rand_score(labels_data, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels_data,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


bench_k_means(model, 'KMeans++', features_data)