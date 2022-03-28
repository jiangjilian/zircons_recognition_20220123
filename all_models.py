# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 00:29:44 2021

@author: 郑宗源
"""
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import os
import las
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from matplotlib import cm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_validate
# from keras.models import Model
# from keras.layers import Dense, Input
# from sklearn_hierarchical_classification import graph
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomTreesEmbedding, ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from pylab import *
from sklearn.model_selection import GridSearchCV

from sklearn.utils import shuffle

np.random.seed(1337)  # for reproducibility


def pca_x(X):
    """
    高斯拟合，进行主成分分析,通过因子变换去除冗余数据
    """
    pca = PCA()
    Xpca = pca.fit_transform(X)
    sns.kdeplot(Xpca[:, 0])
    sns.kdeplot(Xpca[:, 1])
    sns.kdeplot(Xpca[:, 2])
    sns.kdeplot(Xpca[:, 3])
    sns.kdeplot(Xpca[:, 4])
    sns.kdeplot(Xpca[:, 5])
    plt.figure(figsize=(8, 4), dpi=200)
    ax = sns.displot(Xpca[:, 5], kde=True)
    mu = round(np.mean(Xpca), 4)
    sigma = round(np.std(Xpca), 4)
    plt.xlabel(u'主成分1', fontsize=14)
    plt.ylabel(u'频率(%)', fontsize=14)
    plt.show()
    return Xpca


def res_x(X, y):
    """
    通过重采样方法补足数据
    """
    X = PCA().fit_transform(X)
    sm = ADASYN()
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


def hasher_x(X):
    # 哈希特征转换
    hasher = RandomTreesEmbedding(n_estimators=10, random_state=0, max_depth=3)
    X_transformed = hasher.fit_transform(X)
    svd = TruncatedSVD(n_components=50)
    X_reduced = svd.fit_transform(X_transformed)
    return X


def decisiontree(X_train, y_train, X_test, Y_test, X_predict):
    # decision_tree
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    param = {'criterion': ['gini', 'entropy'],
             'max_depth': [5, 10, 20, 40],
             'min_samples_leaf': [2, 3, 5, 10],
             'min_samples_split': [2, 4, 6, 8, 10]
             }
    grid = GridSearchCV(clf, param_grid=param, cv=LeaveOneOut(), scoring='f1', return_train_score=False)
    grid.fit(X_train, y_train)
    print("Best param:", grid.best_params_)
    pred_test = grid.predict(X_test)
    classify = classification_report(Y_test, pred_test)
    print(classify)

    pred = clf.predict(X_predict)

    return pred_test, pred, classify


def my_GaussianNB(X_train, y_train, X_test, Y_test, X_predict):
    """
    Guassian Naive Bayes
    """
    std_clf = GaussianNB()
    std_clf.fit(X_train, y_train)
    eclf = CalibratedClassifierCV(std_clf, method="isotonic", cv="prefit")
    eclf.fit(X_train, y_train)

    pred_test = eclf.predict(X_test)
    classify = classification_report(Y_test, pred_test)
    print(classify)

    pred = eclf.predict(X_predict)

    return pred_test, pred, classify


def mlp(X_train, y_train, X_test, Y_test, X_predict):
    """
    Multilayer Perceptron
    """

    clf = MLPClassifier()
    param = {'hidden_layer_sizes': [(100,), (100, 40), (100, 80, 40)],
             'solver': ['adam', 'sgd', 'lbfgs'],
             'max_iter': [100, 1000],
             'learning_rate_init': [0.001, 0.01, 0.05, 0.1],
             # 'early_stopping': [True]
             }

    grid = GridSearchCV(clf, param_grid=param, cv=LeaveOneOut(), scoring='f1', return_train_score=False)
    grid.fit(X_train, y_train)
    print("Best param:", grid.best_params_)
    pred_test = grid.predict(X_test)
    classify = classification_report(Y_test, pred_test)
    print(classify)

    pred = grid.predict(X_predict)

    return pred_test, pred, classify


def Bernoulli(X_train, y_train, X_test, Y_test, X_predict):
    """
    Bernoulli

    """
    clf = BernoulliNB()
    param = {'alpha':  [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0],
             'fit_prior': [True],
             #'class_prior': [2]
             }

    grid = GridSearchCV(clf, param_grid=param, cv=LeaveOneOut(), scoring='f1')
    grid.fit(X_train, y_train)
    print("Best param:", grid.best_params_)
    pred_test = grid.predict(X_test)
    classify = classification_report(Y_test, pred_test)
    print(classify)

    pred = grid.predict(X_predict)

    return pred_test, pred, classify


def randomtree(X_train, y_train, X_test, Y_test, X_predict):
    clf = RandomForestClassifier()
    param = {'criterion': ['gini', 'entropy'],
             'n_estimators': [10, 100, 200, 400, 1000],
             'max_depth': [5, 10, 20, 40],
             'min_samples_leaf': [2, 3, 5, 10],
             'min_samples_split': [2, 4, 6, 8, 10]
             }

    grid = GridSearchCV(clf, param_grid=param, cv=LeaveOneOut(), scoring='f1')
    #, bootstrap=True, oob_score=True
    grid.fit(X_train, y_train)
    print("Best param:", grid.best_params_)
    pred_test = grid.predict(X_test)
    classify = classification_report(Y_test, pred_test)
    print(classify)

    pred = grid.predict(X_predict)

    return pred_test, pred, classify


def SVM(X_train, y_train, X_test, Y_test, X_predict):
    params = {'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
              'C': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
              'gamma': [0.001, 0.01, 0.1, 1, 2, 4, 8, 10, 100, 1000]
              }
    svc = svm.SVC()
    grid = GridSearchCV(svc, params, cv=LeaveOneOut(), scoring='f1')
    grid.fit(X_train, y_train)
    print("Best param:", grid.best_params_)

    best_params = grid.best_params_
    clf = svm.SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])

    clf.fit(X_train, y_train)
    pred_test = clf.predict(X_test)
    classify = classification_report(Y_test, pred_test)
    print(classify)

    pred = clf.predict(X_predict)

    return pred_test, pred, classify


def KNN(X_train, y_train, X_test, Y_test, X_predict):
    params = {'n_neighbors': [1, 2, 3, 4, 5, 8, 10, 15, 20, 50, 100],
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
              }
    clf = KNeighborsClassifier()
    grid = GridSearchCV(clf, params, cv=LeaveOneOut(), scoring='f1')
    grid.fit(X_train, y_train)
    print("Best param:", grid.best_params_)

    best_params = grid.best_params_
    clf = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'], algorithm=best_params['algorithm'])

    clf.fit(X_train, y_train)
    pred_test = clf.predict(X_test)
    classify = classification_report(Y_test, pred_test)
    print(classify)

    pred = clf.predict(X_predict)

    return pred_test, pred, classify

def logistic(X_train, y_train, X_test, Y_test, X_predict):
    clf = LogisticRegression()
    param = {'penalty': ['l1', 'l2'],
             'C': [0.01, 0.05, 0.1, 1, 2, 4, 8, 10, 100],
             'solver': ['lbfgs', 'sag', 'saga'],
             'multi_class': ['multinomial']
             }

    grid = GridSearchCV(clf, param_grid=param, cv=LeaveOneOut(), scoring='f1')
    # , bootstrap=True, oob_score=True
    grid.fit(X_train, y_train)
    print("Best param:", grid.best_params_)
    pred_test = grid.predict(X_test)
    classify = classification_report(Y_test, pred_test)
    print(classify)

    pred = grid.predict(X_predict)

    return pred_test, pred, classify


def voting(X_train, y_train, X_test, Y_test, X_predict):
    clf1 = LogisticRegression(random_state=1, max_iter=2000)
    clf2 = RandomForestClassifier(random_state=1)

    # priors = 3
    # priors=priors
    clf3 = GaussianNB()
    eclf = VotingClassifier(
        estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
        voting='soft')

    dtc = DecisionTreeClassifier(random_state=1)
    rfc = RandomForestClassifier(random_state=1)
    ada = AdaBoostClassifier(random_state=1)
    gdb = GradientBoostingClassifier(random_state=1)
    eclf = VotingClassifier(estimators=[('dtc', dtc), ('rfc', rfc),
                                        ('ada', ada), ('gdb', gdb)], voting='soft')
    # cv_results = cross_val_score(eclf, X, y, cv=LeaveOneOut())
    # , scoring="f1"
    # print(cv_results)
    params = [{'gdb__n_estimators': [10, 20]}]

    grid_search = GridSearchCV(eclf, params, cv=LeaveOneOut(), scoring='f1')
    # , verbose=3, return_train_score=False
    grid_search.fit(X_train, y_train)
    #print("Best param:", grid.best_params_)

    eclf.fit(X_train, y_train)
    pred_test = eclf.predict(X_test)
    pred = eclf.predict(X_predict)
    classify = classification_report(Y_test, pred_test)
    print(classify)

    return pred_test, pred, classify


def bagging(X_train, y_train, X_test, Y_test, X_predict):
    # bagging
    tree = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=None)
    clf = BaggingClassifier(base_estimator=tree,
                            bootstrap_features=False,
                            n_jobs=-1,
                            random_state=1)

    param = {'n_estimators': [10, 100, 200],
             'max_samples': [0.8, 0.9, 1.0],
             'max_features': [0.8, 0.9, 1.0]
             }

    grid = GridSearchCV(clf, param_grid=param, cv=LeaveOneOut(), scoring='f1')
    #, bootstrap=True, oob_score=True
    grid.fit(X_train, y_train)
    print("Best param:", grid.best_params_)
    pred_test = grid.predict(X_test)
    classify = classification_report(Y_test, pred_test)
    print(classify)

    pred = grid.predict(X_predict)

    return pred_test, pred, classify


def adaboost(X_train, y_train, X_test, Y_test, X_predict):
    clf = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators=1000)
    param = {'n_estimators': [10, 100, 200, 400],
             'learning_rate': [0.1, 0.5, 0.8, 1.0, 2.0],
             }

    grid = GridSearchCV(clf, param_grid=param, cv=LeaveOneOut(), scoring='f1')
    #, bootstrap=True, oob_score=True
    grid.fit(X_train, y_train)
    print("Best param:", grid.best_params_)
    pred_test = grid.predict(X_test)
    classify = classification_report(Y_test, pred_test)
    print(classify)

    pred = grid.predict(X_predict)

    return pred_test, pred, classify


def Vote2(X, y, X0_test, Y_test, X0_predict):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf1 = make_pipeline(MinMaxScaler(), MLPClassifier(max_iter=1000000))
    clf2 = ExtraTreesClassifier()
    clf3 = make_pipeline(GaussianNB())
    # PCA(),
    clf4 = make_pipeline(svm.SVC(probability=True))
    # StandardScaler(),
    eclf = VotingClassifier(
        estimators=[('mlp', clf1), ('rf', clf2), ('gnb', clf3), ('svm', clf4)],
        voting='soft').fit(X_train, y_train)
    eclf.fit(X_train, y_train)
    pred0 = eclf.predict(X_test)
    classify1 = classification_report(y_test, pred0)
    print(classify1)
    eclf.fit(X, y)
    pred_test = eclf.predict(X0_test)
    pred = eclf.predict(X0_predict)
    classify2 = classification_report(Y_test, pred_test)
    print(classify2)
    return pred_test, pred, classify2
