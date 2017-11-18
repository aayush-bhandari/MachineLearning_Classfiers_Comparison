from sklearn import ensemble
from sklearn import  linear_model
from sklearn import  naive_bayes
from sklearn import neural_network
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import metrics
import  numpy as np
import warnings
import pandas as pd
with warnings.catch_warnings():
        # warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    classfiers = [tree.DecisionTreeClassifier(),linear_model.Perceptron(),neural_network.MLPClassifier(),svm.SVC(),naive_bayes.MultinomialNB(),
                  linear_model.LogisticRegression() ,neighbors.KNeighborsClassifier(), ensemble.BaggingClassifier(),ensemble.RandomForestClassifier(),
                  ensemble.AdaBoostClassifier(),ensemble.GradientBoostingClassifier()]

    classfiers_names = ["DecisionTreeClassifier","Perceptron","MLPClassifier","SVC()","MultinomialNB", "LogisticRegression" ,"KNeighborsClassifier",
                        "BaggingClassifier","RandomForestClassifier", "AdaBoostClassifier","GradientBoostingClassifier"]

    accuracy = {}
    precision = {}

    input_dataset = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data",header=None)
    input_dataset[0] = input_dataset[0].astype('category')
    input_dataset[0] = input_dataset[0].cat.codes
    Y = input_dataset.pop(0)
    X = input_dataset

    # Converting dataset of list
    X = X.values.tolist()
    Y = Y.values.tolist()

    # Converting list to numpy array
    X = np.array(X)
    Y = np.array(Y)

    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):
        X_train,X_test = X[train_index],X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        i=0
        for clf in classfiers:
            clf = clf.fit(X_train, Y_train)
            prediction = clf.predict(X_test)
            if not classfiers_names[i] in accuracy:
                accuracy[classfiers_names[i]] =  [accuracy_score(Y_test, prediction)]
                precision[classfiers_names[i]] =  [metrics.precision_score(Y_test,prediction,average='weighted')]
            else:
                accuracy[classfiers_names[i]].append(accuracy_score(Y_test, prediction))
                precision[classfiers_names[i]].append(metrics.precision_score(Y_test, prediction, average='weighted'))
            i = i+1

    # print(accuracy)
    # print(precision)

    avgAccuracy = {}
    avgPrecision = {}
    for key in accuracy:
        avgAccuracy[key] = format(np.mean(accuracy[key]),".2f")
        avgPrecision[key] = format(np.mean(precision[key]),".2f")
    print(avgAccuracy)
    print(avgPrecision)