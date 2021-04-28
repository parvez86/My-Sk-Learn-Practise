from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import make_pipeline
import seaborn as sns
import matplotlib.pyplot as plt
# loading data

iris = datasets.load_iris()
features = iris.data
labels = iris.target

# classes = ['Iris Setosa', 'Iris Versicolour','Iris Virginica']
classes = iris.target_names
features_name = iris.feature_names
print('features: ', features_name)
print('targets: ', classes)
print(features.shape, labels.shape)

# splitting into train test data
train_set, test_set, train_labels, test_labels = train_test_split(features, labels, train_size=0.8)

# Building SVM Model
model = SVC(kernel='linear', tol=1e-3)
model.fit(train_set, train_labels)

model2 = make_pipeline(PolynomialFeatures(degree=3),StandardScaler(),LinearSVC(tol=1e-3,max_iter=10000))
model2.fit(train_set, train_labels)

model3 = NuSVC(kernel='linear',tol=1e-3, shrinking=False)
model3.fit(train_set, train_labels)

# making prediction and evaluating accuracy
predictions = model.predict(test_set)

accuracy = accuracy_score(predictions, test_labels)
print('SVM accuracy: ', accuracy)
print('SVM accuracy: ', accuracy)
print('Linear SVM accuracy: ', accuracy_score(model2.predict(test_set), test_labels))
print('Nu SVM accuracy: ', accuracy_score(model3.predict(test_set), test_labels))

print('SVM co-efficient: ', model.coef_)
print('Linear SVM co-efficient: ', model2['linearsvc'].coef_)
print('Nu SVM co-efficient: ', model3.coef_)
# print('SVM dual co-efficient: ', model.dual_coef_)
print('SVM intercept: ', model.intercept_)
print('Linear SVM intercept: ', model2['linearsvc'].intercept_)
print('Nu SVM intercept: ', model3.intercept_)
# print('SVM fit status: ', model.fit_status_)
# print('SVM classes: ', model.classes_)
# print('SVM support: ', model.support_)
# print('SVM support vectors: ', model.support_vectors_)
# print('SVM # support vectors: ', model.n_support_)
print('predictions: ', predictions)
print('test labels: ', test_labels)

# viewing the result
# for i in range(len(predictions)):
#     print('Predict label: ', classes[predictions[i]], '\tActual label: ', classes[test_labels[i]])

# sns.set()
# sns.pairplot(iris, hue='species', height=3)
# plt.show()