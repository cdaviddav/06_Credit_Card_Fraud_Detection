import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import cross_val_predict
import os
import pandas as pd

from settings import Settings
settings = Settings()

def classification_pipeline(x_train, y_train, pipe, cv, scoring, param_grid):

    cv_ = cv
    scoring_ = scoring
    param_grid_ = param_grid

    reg = GridSearchCV(pipe, cv = cv_, scoring=scoring_, param_grid=param_grid_)
    reg.fit(x_train, y_train)

    # get the explained variance achieved with dimension reduction -> see if # of dimensions can be reduced further
    #reg.best_estimator_.named_steps['reduce_dim'].explained_variance_


    # get scores for regression model
    means = reg.cv_results_['mean_test_score']
    stds = reg.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, reg.cv_results_['params']):
        print(str(reg.scoring) + " Score: " +  "%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print('Best ' + str(reg.scoring) + " Score: " + str(max(means)))
    pipe_best_params = reg.best_params_

    # get best parameter to fit them into the testing model
    return pipe_best_params




def classification_report_csv(report, algo, phase, output_file_path=None):
    df = pd.DataFrame(report).transpose()
    if output_file_path is not None:
        df.to_csv(os.path.join(output_file_path, str(algo) + '_classification_report_' + phase + '.csv'))


def plot_precision_recall_curve(y_train, y_train_scores, algo, phase, output_file_path=None):
    precicions, recalls, _ = precision_recall_curve(y_train, y_train_scores)
    undersample_average_precision = average_precision_score(y_train, y_train_scores)

    fig09 = plt.figure()
    plt.step(recalls, precicions, color='#004a93', alpha=0.2, where='post')
    plt.fill_between(recalls, precicions, step='post', alpha=0.2, color='#48a6ff')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(algo + ' ' + phase + ': \n Average Precision-Recall Score ={0:0.2f}'.format(undersample_average_precision), fontsize=16)
    if output_file_path is not None:
        fig09.savefig(os.path.join(output_file_path, str(algo) + '_precision_recall_curve_' + phase + '.pdf'))


def plot_precision_recall_vs_threshold(y_train, y_train_scores, algo, phase, output_file_path=None):
    precicions, recalls, thresholds = precision_recall_curve(y_train, y_train_scores)
    fig10 = plt.figure()
    plt.plot(thresholds, precicions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g--', label='Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.title(algo + ' ' + phase)
    plt.ylim([0,1])
    if output_file_path is not None:
        fig10.savefig(os.path.join(output_file_path, str(algo) + '_precision_recall_vs_threshold_' + phase + '.pdf'))


def plot_roc_curve(y_train, y_train_scores, algo, phase, output_file_path=None):
    fpr, tpr, thresholds = roc_curve(y_train, y_train_scores)
    fig11 = plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label=None)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.title(algo + ' ' + phase)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if output_file_path is not None:
        fig11.savefig(os.path.join(output_file_path, str(algo) + '_roc_curve_' + phase + '.pdf'))








def evaluate_pipe_best_train(x_train, y_train, pipe_best, algo, binary):
    pipe_best.fit(x_train, y_train)
    y_train_predict = pipe_best.predict(x_train)
    print(classification_report(y_train, y_train_predict))
    print('Accuracy Train: {}'.format(accuracy_score(y_train, y_train_predict)))


    report = classification_report(y_train, y_train_predict, output_dict=True)
    classification_report_csv(report, algo, 'train', output_file_path=settings.csv)


    if algo in ("RandomForestClassifier", "KNeighborsClassifier", 'KNearest'): # if algo has no decision function
        y_train_proba = cross_val_predict(pipe_best, x_train, y_train, cv=5, method="predict_proba")
        y_train_scores = y_train_proba[:, 1]

    else:
        y_train_scores = cross_val_predict(pipe_best, x_train, y_train, cv=5, method="decision_function")


    if binary == True:
        plot_precision_recall_vs_threshold(y_train, y_train_scores, algo, 'train', output_file_path=settings.figures)
        plot_roc_curve(y_train, y_train_scores, algo, 'train', output_file_path=settings.figures)
        plot_precision_recall_curve(y_train, y_train_scores, algo, 'train', output_file_path=settings.figures)




def evaluate_pipe_best_test(x_test, y_test, pipe_best, algo, binary):
    y_test_predict = pipe_best.predict(x_test)
    print(classification_report(y_test, y_test_predict))
    print('Accuracy Test: {}'.format(accuracy_score(y_test, y_test_predict)))

    report = classification_report(y_test, y_test_predict, output_dict=True)
    classification_report_csv(report, algo, 'test', output_file_path=settings.csv)


    if algo in ("RandomForestClassifier", "KNeighborsClassifier", 'KNearest'): # if algo has no decision function
        y_test_proba = cross_val_predict(pipe_best, x_test, y_test, cv=5, method="predict_proba")
        y_test_scores = y_test_proba[:, 1]

    else:
        y_test_scores = cross_val_predict(pipe_best, x_test, y_test, cv=5, method="decision_function")


    if binary == True:
        plot_precision_recall_vs_threshold(y_test, y_test_scores, algo, 'test', output_file_path=settings.figures)
        plot_roc_curve(y_test, y_test_scores, algo, 'test', output_file_path=settings.figures)
        plot_precision_recall_curve(y_test, y_test_scores, algo, 'test', output_file_path=settings.figures)