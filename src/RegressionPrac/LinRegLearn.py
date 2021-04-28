from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

# loading the boston house pricing data
data = datasets.load_boston()

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

# plotting the datasets
def show_plot(features, labels, predictions = []):
    target_name = 'price($)'
    if len(predictions)!=0:
        plt.scatter(test_labels, predictions, edgecolors=(0, 0, 0))
        plt.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'k--', lw=4)
        plt.xlabel(f'Measured{target_name}')
        plt.ylabel(f'Predicted{target_name}')
        plt.show()

    else:
        for i in range(13):
            features_name = data.feature_names
            plt.scatter(features.T[i], labels)
            plt.xlabel(features_name[i])
            plt.ylabel(target_name)
            plt.show()


# show_plot(features,labels)
# Building the model
model = LinearRegression()
model.fit(train_set, train_labels)

# making predictions and getting accuracy
predictions = model.predict(test_set)

print('Linear Regression score: ',model.score(features, labels))
print('Model intercept: ', model.intercept_)
print('Model co_efficients: ', model.coef_)
# show_plot(test_set, test_labels, predictions)

xfit = np.linspace(1, 22,13).reshape(1, -1)
print(xfit)
print(test_set[0])
print(xfit.shape)
# Xfit = xfit[:, np.newaxis]
print([12] * 13)
yfit = model.predict(xfit)
print('Prediction: ', yfit)
