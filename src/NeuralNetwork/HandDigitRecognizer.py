from PIL import Image
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import mnist
import joblib
# trainning variables
x_train = mnist.train_images()
y_train = mnist.train_labels()

print(type(x_train))
print(x_train.shape)
# testing variables
x_test = mnist.test_images()
y_test = mnist.test_labels()

# check data
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# reshaping the data
x_train = x_train.reshape((-1, 28 * 28))
x_test = x_test.reshape((-1, 28 * 28))
x_train = (x_train/256)
x_test = (x_test/256)
print(x_test.shape)

# Building the model
model = MLPClassifier(solver='adam', hidden_layer_sizes=(64, 64))
model.fit(x_train, y_train)

predictions = model.predict(x_test)
conf_matrix = confusion_matrix(y_test, predictions)

def model_accuracy(conf_matrix):
    diagonal_value = conf_matrix.trace()
    n_element = conf_matrix.sum()
    return diagonal_value/n_element

accuracy = model_accuracy(conf_matrix)
print('Model accuracy: ', accuracy)

joblib.dump(model, 'HandDigitRecognizer.joblib')