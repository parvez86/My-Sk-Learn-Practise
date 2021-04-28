import numpy as np
from sklearn import preprocessing
data = [
   [2.1, -1.9, 5.5],
   [-1.5, 2.4, 3.5],
   [0.5, -7.9, 5.6],
   [5.9, 2.3, -5.8]
]

# Binarize the input data
binarized = preprocessing.Binarizer(threshold=0.5).fit_transform(data)
print("\nBinarized data:\n", binarized)

#displaying the mean and the standard deviation of the input data
print("Mean =", np.array(data).mean(axis=0))
print("Stddeviation = ", np.array(data).std(axis=0))

#Removing the mean and the standard deviation of the input data
data_scaled = preprocessing.scale(data)
print("Mean_removed =", data_scaled.mean(axis=0))
print("Stddeviation_removed =", data_scaled.std(axis=0))

print(data_scaled)

data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = data_scaler_minmax.fit_transform(data)
print ("\nMin max scaled data:\n", data_scaled_minmax)

# Normalization
data_normalized_l1 = preprocessing.normalize(data, norm='l1')
print("\nL1 normalized data:\n", data_normalized_l1)

data_normalized_l2 = preprocessing.normalize(data, norm='l2')
print("\nL1 normalized data:\n", data_normalized_l2)