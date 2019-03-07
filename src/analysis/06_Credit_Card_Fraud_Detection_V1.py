
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 17:40:39 2017

@author: cdavid
"""

""" Import all packages and the used settings and functions """

import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')


from settings import Settings
from src.functions.data_preprocessing import *
from src.functions.functions_plot import *
from src.functions.classification_pipeline import *

settings = Settings()


""" ---------------------------------------------------------------------------------------------------------------
Load training and test dataset
"""

# Load train and test dataset
df = pd.read_csv(settings.config['Data Locations'].get('creditcard'))
#df = df[:10000]
df.name = 'df'


# Target variable: Class


""" ---------------------------------------------------------------------------------------------------------------
Explore the data
First take a look at the training dataset
- what are the features and how many features does the training data include
- are the missing values (but take a deeper look at the data preperation process)
- what are the different units of the features
"""

def data_exploration(df):

    # Get a report of the training and test dataset as csv
    # -> Use the function describe_report(df, name, output_file_path=None)
    describe_report(df, output_file_path=settings.csv)


    # Create boxplots to indentify outliers. Histograms are a good standard way to see if feature is skewed but to find outliers, boxplots are the way to use
    # -> Use the function create_boxplots(df, output_file_path=None)
    create_boxplots(df, output_file_path=settings.figures)

    # Check if target variable is balanced
    sns.countplot('Class', data=df)


    #Time comparse between fraud and normal transactions
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))
    bins = 50
    ax1.hist(df.Time[df.Class == 1], bins = bins)
    ax1.set_title('Fraud')
    ax2.hist(df.Time[df.Class == 0], bins = bins)
    ax2.set_title('Normal')
    plt.xlabel('Time (in Seconds)')
    plt.ylabel('Number of Transactions')
    plt.show()


    #Amount comparse between fraud and normal transactions
    print('Median amount of Fraud transcation: {}'.format(df.Amount[df.Class == 1].median()))
    print('Mean amount of Fraud transcation: ' + '%0.2f' % (df.Amount[df.Class == 1].mean()))
    print('90% quantil amount of Fraud transcation: ' + '%0.2f' % (np.percentile(df.Amount[df.Class == 1], 90)))
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))
    bins = 30
    ax1.hist(df.Amount[df.Class == 1], bins = bins)
    ax1.set_title('Fraud')
    ax2.hist(df.Amount[df.Class == 0], bins = bins)
    ax2.set_title('Normal')
    plt.xlabel('Amount ($)')
    plt.ylabel('Number of Transactions')
    plt.yscale('log')
    plt.show()



    #Compare time and amount between fraud and normal transactions
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,6))
    ax1.scatter(df.Time[df.Class == 1], df.Amount[df.Class == 1])
    ax1.set_title('Fraud')
    ax2.scatter(df.Time[df.Class == 0], df.Amount[df.Class == 0])
    ax2.set_title('Normal')
    plt.xlabel('Time (in Seconds)')
    plt.ylabel('Amount')
    plt.show()


    #Analysis of anonymized features
    v_features = df.iloc[:,1:29].columns #Select only the anonymized features.
    for v_feature in v_features:
        fig, ax = plt.subplots()
        sns.distplot(df[v_feature][df.Class == 1], bins=50, color='r')
        sns.distplot(df[v_feature][df.Class == 0], bins=50, color ='b')
        ax.set_xlabel('')
        ax.set_title('histogram of feature: ' + str(v_feature))
        fig.savefig(os.path.join(settings.figures, 'distplot_' + str(v_feature) + '.pdf'))

#data_exploration(df)

#Drop all of the features that have very similar distributions between the two types of transactions.
#Because when distributions are similar, the algorithmus can not learn differences from distributions.
df = df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)

""" ---------------------------------------------------------------------------------------------------------------
Feature Creation
"""

# Split dataset before using undersampling or oversampling to test on original dataset
from sklearn.model_selection import StratifiedShuffleSplit
print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')


sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
#for train_index, test_index in sss.split(X, y):
for train_index, test_index in sss.split(df, df['Class']):
    strat_train = df.loc[train_index]
    strat_test = df.loc[test_index]

print('Frauds', round(strat_train['Class'].sum()/strat_train['Class'].count()*100,2), '% after StratifiedShuffleSplit in training set')
print('Frauds', round(strat_test['Class'].sum()/strat_test['Class'].count()*100,2), '% after StratifiedShuffleSplit in training set')



count_Normal_transacation = len(strat_train[strat_train["Class"]==0]) # normal transaction are repersented by 0
count_Fraud_transacation = len(strat_train[strat_train["Class"]==1]) # fraud by 1

normal_indices = np.array(strat_train[strat_train.Class==0].index)
fraud_indices= np.array(strat_train[strat_train.Class==1].index)

normal_data = strat_train[strat_train["Class"]==0]
fraud_data = strat_train[strat_train["Class"]==1]



def undersample(normal_indices, fraud_indices, times):  # times denote the normal data = times*fraud data
    Normal_indices_undersample = np.array(
        np.random.choice(normal_indices, (times * count_Fraud_transacation), replace=False))
    undersample_data = np.concatenate([fraud_indices, Normal_indices_undersample])
    undersample_data = df.iloc[undersample_data, :]

    print('the normal transaction proportion is : {0:0.4f}'.format(
        len(undersample_data[undersample_data.Class == 0]) / undersample_data.shape[0]))
    print('the fraud transaction proportion is : {0:0.4f}'.format(
        len(undersample_data[undersample_data.Class == 1]) / undersample_data.shape[0]))
    print('total number of record in resampled data is:' + str(undersample_data.shape[0]))
    return (undersample_data)

df_train = undersample(normal_indices, fraud_indices, 1)




""" ---------------------------------------------------------------------------------------------------------------
Feature Selection
"""


# Outlier removal via Interquartile Range Method
def remove_outlier_Interquartile_Range_Method(df_train, feature):
    print(feature)
    v_fraud = df_train[feature].loc[df_train['Class'] == 1].values
    q25, q75 = np.percentile(v_fraud, 25), np.percentile(v_fraud, 75)
    print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
    v14_iqr = q75 - q25
    print('iqr: {}'.format(v14_iqr))

    v_cut_off = v14_iqr * 1.5
    v_lower, v_upper = q25 - v_cut_off, q75 + v_cut_off
    print('Cut Off: {}'.format(v_cut_off))
    print('Lower: {}'.format(v_lower))
    print('Upper: {}'.format(v_upper))

    outliers = [x for x in v_fraud if x < v_lower or x > v_upper]
    print('Feature Outliers for Fraud Cases: {}'.format(len(outliers)))
    print('Outliers:{}'.format(outliers))

    new_df = df_train.drop(df_train[(df_train[feature] > v_upper) | (df_train[feature] < v_lower)].index)
    print('Number of Instances after outliers removal: {}'.format(len(new_df)))
    print('----' * 20)
    return new_df

feature_list = list(df_train)
feature_list.remove('Class')

for feature in feature_list:
    df_train = remove_outlier_Interquartile_Range_Method(df_train, feature)




#Dimensionality Reduction and Clustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD

X = df_train.drop('Class', axis=1)
y = df_train['Class']

X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values) # T-SNE Implementation
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values) # PCA Implementation
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)



f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
# labels = ['No Fraud', 'Fraud']
f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)

import matplotlib.patches as mpatches
blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')


# t-SNE scatter plot
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax1.set_title('t-SNE', fontsize=14)
ax1.grid(True)
ax1.legend(handles=[blue_patch, red_patch])


# PCA scatter plot
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax2.set_title('PCA', fontsize=14)
ax2.grid(True)
ax2.legend(handles=[blue_patch, red_patch])

# TruncatedSVD scatter plot
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)

ax3.grid(True)
ax3.legend(handles=[blue_patch, red_patch])
plt.show()




""" ---------------------------------------------------------------------------------------------------------------
Machine Learning (Classification)
"""

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df_train, test_size=0.2, random_state=42)

y_train = y
x_train = X

y_test = strat_test['Class']
x_test = strat_test.drop(['Class'], axis=1)


pipeline = Pipeline([
    ('reduce_dim', PCA()),
    ('feature_scaling', MinMaxScaler()), # scaling because linear models are sensitive to the scale of input features
    ('classification', RandomForestClassifier()),
    ])

param_grid = [{'reduce_dim__n_components': [5, 15, 19],
               'classification__max_depth': [4, 9, 20, 50, 100],
               'classification__verbose': [0, 1]
              }]


pipe_best_params = classification_pipeline(x_train, y_train, pipeline, 5, 'accuracy', param_grid)

pipe_best = Pipeline([
    ('reduce_dim', PCA(n_components = pipe_best_params['reduce_dim__n_components'])),
    ('feature_scaling', MinMaxScaler()),
    ('classification', RandomForestClassifier(
        max_depth = pipe_best_params['classification__max_depth'],
        verbose = pipe_best_params['classification__verbose'],))
])

print(pipe_best_params['reduce_dim__n_components'])
print(pipe_best_params['classification__max_depth'])
print(pipe_best_params['classification__verbose'])

train_errors = evaluate_pipe_best_train(x_train, y_train, pipe_best, 'RandomForestClassifier', binary=True)


plot_learning_curve(pipe_best, 'RandomForestClassifier', x_train, y_train, 'accuracy', output_file_path=settings.figures)



""" ---------------------------------------------------------------------------------------------------------------
Evaluate the System on the Test Set
"""
#Evaluate the model with the test_set
# -> Use the function evaluate_pipe_best_test(x_train, y_train, pipe_best, algo, output_file_path=None)
evaluate_pipe_best_test(x_test, y_test, pipe_best)
















#https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/notebook














