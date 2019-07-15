
''' 
HOW TO RUN :-
python3 validation_test.py /Users/shehjarsadhu/Desktop/KAYA_DataTest/ML/Datasets/Day1/twofeatures_10patients.csv /Users/shehjarsadhu/Desktop/KAYA_DataTest/ML/Plots
'''

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve


def plot_learning_curves(X, y,save_to):
    train_sizes, train_scores, valid_scores = learning_curve(
       SVC(kernel='rbf',C= 10,gamma= 0.0001,max_iter =-1,decision_function_shape = "ovo") , X, y,train_sizes=[4,3,5]
, cv=2)
        # SVC(kernel='linear',C= 10, gamma= 0.0001)
        # KNeighborsClassifier(algorithm='kd_tree', n_neighbors= 3, p= 2)
    print("train_sizes = ",train_sizes)
    print("train_scores = ",train_scores)
    print("valid_scores = ",valid_scores)
    
    plt.subplot(2,1,1)
    plt.grid()
    plt.title("Learning Curves \n {kernel='linear',C= 10, gamma= 0.0001}")
    plt.xlabel("Training examples \n \n ")
    plt.ylabel("Accuracy")
    plt.plot(train_sizes,  np.mean(train_scores,axis =1), 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, np.mean(valid_scores,axis =1), 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.grid()
    plt.subplot(2,1,2)
    plt.xlabel("Training examples")
    plt.ylabel("Error")
    plt.plot(train_sizes,  1-np.mean(train_scores,axis =1), 'o-', color="r",
             label="Training error")
    plt.plot(train_sizes, 1-np.mean(valid_scores,axis =1), 'o-', color="g",
             label="Cross-validation error")
    plt.legend(loc="best")
    plt.savefig(save_to + "/" + "learning_curves")
    plt.show()

def main():
    dataset = sys.argv[1]
    save_to = sys.argv[2]
    print("sys.argv[2] = ",dataset)
    df = pd.read_csv(dataset)
    X = df.drop(['ExerciseNames'], axis=1)
    y =  target = df["ExerciseNames"]
    train_sizes=[4,3,5]
    plot_learning_curves(X, y,save_to)


if __name__== "__main__":
    main()