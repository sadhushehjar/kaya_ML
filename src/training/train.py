'''
HOW TO RUN :-
python3 train.py /Users/shehjarsadhu/Desktop/KAYA/ML/Datasets/ClassificationData/Day1/twofeatures_10patients.csv /Users/shehjarsadhu/Desktop/KAYA/ML/ModelSelection/Hyperparameters/kaya_svm.json /Users/shehjarsadhu/Desktop/KAYA/ML/ModelSelection/Output/svm_output.txt /Users/shehjarsadhu/Desktop/KAYA/ML/ModelSelection/Hyperparameters/kaya_knn.json /Users/shehjarsadhu/Desktop/KAYA/ML/ModelSelection/Output/knn_output.txt /Users/shehjarsadhu/Desktop/KAYA/ML/ModelSelection/Hyperparameters/kaya_logisticreg.json /Users/shehjarsadhu/Desktop/KAYA/ML/ModelSelection/Output/lr_output.txt /Users/shehjarsadhu/Desktop/KAYA/ML/ModelSelection/Hyperparameters/kaya_decisionTree.json /Users/shehjarsadhu/Desktop/KAYA/ML/ModelSelection/Output/decision_tree_output.txt
'''

import os
import csv
import sys
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Uses sklearns support vector machine and retruns a list pf predictions.
def support_vector_machine(X,y,X_train, y_train,X_test,y_test,df_json,output_path):
    SVM = svm.SVC( decision_function_shape='ovo', verbose=1)
    print("params = ", df_json["param_grid"])
    clf = GridSearchCV(SVM, df_json["param_grid"], cv=2,
                       scoring='accuracy')
    clf.fit(X_train, y_train)
    
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    f = open(output_path,'a')

    means_list = []
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("mean cross-validation accuracy: ", mean, "for", params)
        f.write("mean cross-validation accuracy: \n")
        f.write(str(mean))
        f.write("\n Hyper Parameters \n")
        f.write(str(params))
        f.write("\n")
        means_list.append(mean)
        print()
    f.write("\nBest Score: \n")
    f.write(str(clf.best_score_))
    f.write("\nBest parameters set found on development set:\n ")
    f.write(str(clf.best_params_))
    print("Best Score: ", clf.best_score_)
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    y_predict = clf.predict(X_test)
    #Classification results.
    print("Confusion matrix for the classifier: \n")
    labels = ["Hands Hold Out","Finger to Nose","Finger Tap","Closed Grip","Hand Flip"]
    cm =confusion_matrix(y_test,y_predict)
    print(cm)

    print("Classification report \n")
    cl_report = classification_report(y_test, y_predict)
    print(cl_report)

    f.write("\n Confusion Matrix \n")
    f.write(str(cm))
    f.write("\n Classification report \n")
    f.write(str(cl_report))
    
    return clf.best_score_

# Runs sklearns k nearest neighbours and retruns a list of pridections.
# Does tuning hyperparameters such as K and performing grid search on that.
def classifierKNN(data,target,X_train, y_train, X_test,y_test,lables,df_json,output_path_knn):
    print("Tuning parameters for: ")
    print(df_json["param_grid"],type(df_json["param_grid"]))
    
    knn = KNeighborsClassifier()

    # Cross validation of hyperparameters.
    clf = GridSearchCV(knn, df_json["param_grid"], cv=2,
                       scoring='accuracy')

    clf.fit(X_train, y_train)

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    f = open(output_path_knn,'a')

    means_list = []
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("mean cross-validation accuracy: ", mean, "for", params)
        f.write("mean cross-validation accuracy: \n")
        f.write(str(mean))
        f.write("\n Hyper Parameters \n")
        f.write(str(params))
        f.write("\n")
        means_list.append(mean)
        print()
    f.write("\nBest Score: \n")
    f.write(str(clf.best_score_))
    f.write("\nBest parameters set found on development set:\n ")
    f.write(str(clf.best_params_))
    print("Best Score: ", clf.best_score_)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    y_predict = clf.predict(X_test)

    # Classification results.
    print("Confusion matrix for the classifier: \n")
    labels = "Hands Hold Out ,Finger to Nose ,Finger Tap, Closed Grip, Hand Flip"
    cm = confusion_matrix(y_test,y_predict)
    print(confusion_matrix(y_test,y_predict))

    print("Classification report \n")
    cl_report = classification_report(y_test, y_predict)
    print(classification_report(y_test, y_predict))
    
    f.write("\n Confusion Matrix \n")
    f.write(str(cm))
    f.write("\n Classification report \n")
    f.write(str(cl_report))
  
    return clf.best_score_

def logistic_regression(X_train,y_train,X_test,y_test,kaya_logistic,output_path_lr):

    clf = GridSearchCV(LogisticRegression(), kaya_logistic["param_grid"], cv=2, n_jobs=-1)
    clf.fit(X_train, y_train)

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    f = open(output_path_lr,'a')

    #lr_output.txt
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("mean cross-validation accuracy: ", mean, "for", params)
        f.write("mean cross-validation accuracy: \n")
        f.write(str(mean))
        f.write("\n Hyper Parameters \n")
        f.write(str(params))
        f.write("\n")
        print()
    f.write("\nBest Score: \n")
    f.write(str(clf.best_score_))
    f.write("\nBest parameters set found on development set:\n ")
    f.write(str(clf.best_params_))
    print("Best Score: ", clf.best_score_)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    y_predict = clf.predict(X_test)

    #Classification results.
    print("Confusion matrix for the classifier: \n")
    labels = ["Hands Hold Out","Finger to Nose","Finger Tap","Closed Grip","Hand Flip"]
    cm = confusion_matrix(y_test,y_predict,labels)
    print(cm)

    print("Classification report \n")
    cl_report = classification_report(y_test, y_predict, target_names=labels)
    print(cl_report)

    f.write("\n Confusion Matrix \n")
    f.write(str(cm))
    f.write("\n Classification report \n")
    f.write(str(cl_report))
    return clf.best_score_

def decision_trees(X_train, y_train, X_test,y_test,df_json,output_path_dt):
    dts = DecisionTreeClassifier(random_state=0)
    clf = GridSearchCV(dts,max_depth=3, cv=2,
                       scoring='accuracy')

    clf.fit(X_train, y_train)

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    f = open(output_path_dt,'a')

    means_list = []
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("mean cross-validation accuracy: ", mean, "for", params)
        f.write("mean cross-validation accuracy: \n")
        f.write(str(mean))
        f.write("\n Hyper Parameters \n")
        f.write(str(params))
        f.write("\n")
        means_list.append(mean)
        print()
    f.write("\nBest Score: \n")
    f.write(str(clf.best_score_))
    f.write("\nBest parameters set found on development set:\n ")
    f.write(str(clf.best_params_))
    print("Best Score: ", clf.best_score_)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    y_predict = clf.predict(X_test)


    #Classification results.
    print("Confusion matrix for the classifier: \n")
    labels = ["Hands Hold Out","Finger to Nose","Finger Tap","Closed Grip","Hand Flip"]
    cm = confusion_matrix(y_test,y_predict,labels)
    print(cm)

    print("Classification report \n")
    cl_report = classification_report(y_test, y_predict, target_names=labels)
    print(cl_report)

    f.write("\n Confusion Matrix \n")
    f.write(str(cm))
    f.write("\n Classification report \n")
    f.write(str(cl_report))

    return clf.best_score_
    

def main():

    dataset = sys.argv[1]
    hyparams_svm = sys.argv[2]
    output_path_svm = sys.argv[3]
    hyparams_knn = sys.argv[4]
    output_path_knn = sys.argv[5]
    hyparams_lr = sys.argv[6]
    output_path_lr = sys.argv[7]
    hyparams_dt = sys.argv[8]
    output_path_dt = sys.argv[9]
    print("\n",sys.argv[0],"\n",sys.argv[1],"\n",sys.argv[2],"\n",sys.argv[3],"\n",sys.argv[4])

    # Load dataset
    df = pd.read_csv(dataset)
    print(df.head())
    print("ExerciseNames ",df["ExerciseNames"])
    data = df.drop(['ExerciseNames'], axis=1)
    lables  = ['Hands Hold Out','Finger to Nose','Finger Tap','Closed Grip','Hand Flip']
    target = df["ExerciseNames"] #['Hands Hold Out','Finger to Nose','Finger Tap','Closed Grip','Hand Flip']
    
    
    print(data.head())
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(data,target , test_size=0.2,shuffle = True) # 80% training and 20% test
   
    print("X_train = ",X_train)
    print("y_train = ",y_train)
    print("X_test = ",X_test)
    print("y_test = ",y_test)

    knn_hyParameters = "/Users/shehjarsadhu/Desktop/KAYA_DataTest/kaya_knn.json"
    with open(hyparams_knn, 'r') as f:
        df_json = json.load(f)
        print("df_json type = ",type(df_json))
    
    #svm_hyParameters = "/Users/shehjarsadhu/Desktop/KAYA_DataTest/kaya_svm.json"
    with open(hyparams_svm, 'r') as f:
        df_json_svm = json.load(f)
        print("df_json type = ",type(df_json_svm))
    
    kaya_logistic = "/Users/shehjarsadhu/Desktop/KAYA_DataTest/kaya_logisticreg.json"
    with open(hyparams_lr, 'r') as f:
        df_json_log = json.load(f)
        print("df_json type = ",type(df_json_log))
    
    #a = df_json["param_grid"]["n_neighbors"]
    #print("df_json = ",a)

    neighs = []
    for i in range(20):
        print(i+1)
        neighs.append(i)
    

    # Where to get accuracies for each feature table?
    # print("param_grid.keys = ",param_grid.keys)
    print("---------------------------------------------------------K-nearest-neighbour------------------------------------------------------------v")
    bestAccuracy_knn = classifierKNN(data,target,X_train, y_train, X_test,y_test,lables,df_json,output_path_knn)

    print("---------------------------------------------------------Support Vector Machine------------------------------------------------------------v")
    bestAccuracy_svm = support_vector_machine(data,target,X_train, y_train,X_test,y_test,df_json_svm,output_path_svm)
    
   # print("---------------------------------------------------------Decision Trees------------------------------------------------------------v")
   # decision_trees(X_train, y_train, X_test,y_test,df_json,output_path_dt)
    print("---------------------------------------------------------Logistic Regression------------------------------------------------------------v")
    bestAccuracy_lr = logistic_regression(X_train,y_train,X_test,y_test,df_json_log,output_path_lr)

if __name__== "__main__":
    main()