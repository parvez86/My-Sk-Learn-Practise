from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# loading the boston house pricing data
data = fetch_openml('mnist_784')

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
model = make_pipeline(StandardScaler(), PCA())

model.fit_transform(train_set)

# making predictions and getting accuracy
predictions = model.transform(test_set)

print('PCA score: ', model.score(features, labels))
# print('PCA mean: ',model['pca'].mean_)
print('Model # components: ', model['pca'].n_components_)
# print('Model components: ', model['pca'].components_)
print('Model # samples: ', model['pca'].n_samples_)
print('Model # features: ', model['pca'].n_features_)
# print('Model explained variance ratio: ', model['pca'].explained_variance_ratio_)
# print('Model explained variance: ', model['pca'].explained_variance_)
print('Model noise variance: ', model['pca'].noise_variance_)
# print('Model singular values: ', model['pca'].singular_values_)


# viewing the result
# for i in range(len(predictions)):
#     print('Predict label: ', data.target_names[predictions[i]], '\tActual label: ', data.target_names[test_labels[i]])
# data['PCA1'] = predictions[:, 0]
# data['PCA2'] = predictions[:, 1]
# sns.lmplot("PCA1", "PCA2", hue='species', data=data, fit_reg=False)
# plt.show()