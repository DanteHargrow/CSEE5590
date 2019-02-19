from sklearn import datasets,svm
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()

#cross validation
xTrain, xTest, yTrain, yTest = train_test_split(iris.data,iris.target,test_size=.4,random_state=0)

#linear and rbf kernels
clf1 = svm.SVC(kernel='rbf',C=1).fit(xTrain,yTrain)
clf = svm.SVC(kernel='linear',C=1).fit(xTrain,yTrain)
modelGNB = GaussianNB()
#naive bayes
modelGNB.fit(xTrain, yTrain)

print("Accuracy of naive bayes on test set: {:.2f}".format(modelGNB.score(xTest,yTest)))
print("Accuracy of SVM linear on test set: {:.2f}".format(clf.score(xTest,yTest)))
print("Accuracy of SVM rbf on test set: {:.2f}".format(clf1.score(xTest,yTest)))



