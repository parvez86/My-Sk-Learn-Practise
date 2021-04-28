from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
print('features: ', data.feature_names)
print('targets: ', data.target_names)

train, test, train_labels, test_labels = train_test_split(
   features,labels,test_size = 0.20, random_state = 42
)
# Gaussian Naive Bayes Classfier Model
GNBclf = GaussianNB()
model = GNBclf.fit(train, train_labels)

# Bernoulli Naïve Bayes classifier model (for text classification with bag of words)
gnb_born_clf = BernoulliNB()
gnb_born_clf.fit(train, train_labels)

preds = GNBclf.predict(test)

print('For GNB classifier: ')
print('Accuracy: ', GNBclf.score(test, test_labels))
print('Probabiiity: ', GNBclf.predict_proba(test))
print('# of training sample in each class: ', GNBclf.class_count_)
print('probability of each class: ', GNBclf.class_prior_)
print('class labels: ', GNBclf.classes_)
print('absolute additive value to variances: ', GNBclf.epsilon_)
print('variance of each feature per class: ', GNBclf.sigma_)
print('mean of each feature per class: ', GNBclf.theta_)


print('\n BernouliNB Classifier: ')
print('Accuracy: ',gnb_born_clf.score(test, test_labels))
# print('Probabiiity: ', gnb_born_clf.predict_proba(test))