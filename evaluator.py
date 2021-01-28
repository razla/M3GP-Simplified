from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, roc_auc_score, mean_squared_error, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.datasets import fetch_openml
from pmlb import fetch_data, classification_dataset_names
from mlxtend.evaluate import permutation_test
from sklearn.metrics import mean_squared_error
from statistics import mean, stdev, median
from string import ascii_lowercase
from operator import itemgetter
from time import process_time
from sys import argv, stdin
from decimal import Decimal
from random import choices
from os.path import exists
from os import makedirs
from pathlib import Path
from scipy.stats import entropy
from m3gp.M3GP import M3GP
from m3gp.Constants import *
import openml

# from os import makedirs
import pandas as pd
import numpy as np
import os

REGULAR = True

USAGE = 'python ' + os.path.basename(__file__) + ' n_replicates dataset_name'

num_of_classes = 0

def rand_str(n): return ''.join(choices(ascii_lowercase, k=n))

def get_args():
    if len(argv) == 3:
        n_replicates, dataset_name = int(argv[1]), str(argv[2])
    else:
        exit('-'*80 + '\n' + 'Incorrect usage: python ' + ' '.join(argv) + '\n' + 'Please use: ' + USAGE + '\n' + '-'*80)
    if not exists('results/'): makedirs('results/')
    res_file = 'results/' + dataset_name + '_' + rand_str(3) + '.txt'
    return res_file, n_replicates, dataset_name

def fprint(f_name, s):
    if stdin.isatty(): print(s)
    with open(Path(f_name), 'a') as f: f.write(s)

def print_params(res_file, n_replicates, dataset_name, dataset_size, num_of_classes, num_of_features):
    fprint(res_file,\
        'Number of replicates: ' + str(n_replicates) + '\n' +\
        'Dataset name: ' + str(dataset_name) + '\n' +\
        'Number of records: ' + str(dataset_size) + '\n' +\
        'Number of classes: ' + str(num_of_classes) + '\n' +\
        'Number of features: ' + str(num_of_features) + '\n')


def get_regular():
    return config.regular

def main():
    global num_of_classes
    res_file, n_replicates, dataset_name = get_args()
    flag = False
    if dataset_name.isnumeric():
        flag = True
        dataset = fetch_openml(data_id=dataset_name, as_frame=True)
        print(dataset.frame)
        num_of_features, dataset_size, num_of_classes = dataset.frame.shape[1]-1, dataset.frame.shape[0], dataset.frame.iloc[:,-1].nunique()
    else:
        dataset = fetch_data(dataset_name)
        num_of_features, dataset_size, num_of_classes = dataset.shape[1]-1, dataset.shape[0], dataset['target'].nunique()

    
    print_params(res_file, n_replicates, dataset_name, dataset_size, num_of_classes, num_of_features)
    fitnesses_name = {0: 'Accuracy', 1: 'Balanced Accuracy', 2: 'Entropy', 3: 'Entropy With Diagonal'}
    fprint(res_file, '\n')
    allreps_bala_acc = {0: [], 1: [], 2: [], 3: []}
    allreps_f1 = {0: [], 1: [], 2: [], 3: []}
    alltimes = {0: [], 1: [], 2: [], 3: []}
    for rep in range(1, n_replicates+1):
        onerep_bala_acc = {0: [], 1: [], 2: [], 3: []}
        onerep_f1 = {0: [], 1: [], 2: [], 3: []}
        # kf = KFold(n_splits = 5, shuffle = True)
        # for train_index, test_index in kf.split(dataset):
        if flag:
            class_header = dataset.frame.columns[-1]
            X_train, X_test, y_train, y_test = train_test_split(dataset.frame.drop(columns=[class_header]), dataset.frame[class_header], train_size=TRAIN_FRACTION)
        else:
            class_header = dataset.columns[-1]
            X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns=[class_header]), dataset[class_header], train_size=TRAIN_FRACTION)
		

        for i in range(4):
            ################### CHANGE ###################
            # X_train, X_test = dataset.loc[train_index, dataset.columns != 'target'], dataset.loc[test_index, dataset.columns != 'target']
            # y_train, y_test = dataset.loc[train_index, dataset.columns == 'target'], dataset.loc[test_index, dataset.columns == 'target']
            model = M3GP(fitnesses_name[i])
            modeler_start_time = process_time()
            model.fit(X_train, y_train)
            alltimes[i].append(process_time() - modeler_start_time)
            predictions = model.predict(X_test)
            predictions = [str(x) for x in predictions]
            y_test = [str(y) for y in y_test]
            # predicitons = [float(x) for x in model.predict(X_test)]
            bala_acc = balanced_accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average="micro") 
            onerep_bala_acc[i].append(bala_acc)
            onerep_f1[i].append(f1)
            allreps_bala_acc[i].append(bala_acc)
            allreps_f1[i].append(f1)


        # stats for one replicate
        s = 'replicate ' + str(rep) + ':\n'
        for i in range(4):
            s += '\t' + fitnesses_name[i] + ' - (Balanced Accuracy median) - (' + str(round(median(onerep_bala_acc[i]), 5)) + ')\n'
            s += '\t' + fitnesses_name[i] + ' - (F1 score median) - (' + str(round(median(onerep_f1[i]), 5)) + ')\n'

        fprint(res_file, s + '\n')

    rankings_medians_bala_acc = {0: 0, 1: 0, 2: 0, 3: 0}
    bala_acc_medians = []
    for i in range(4):
        bala_acc_medians.append([list(rankings_medians_bala_acc.keys())[i], median(allreps_bala_acc[i])])
    bala_acc_medians = sorted(bala_acc_medians, key=itemgetter(1), reverse=True)
    names_to_indexes = list(fitnesses_name.values())

    rankings_medians_f1_score = {0: 0, 1: 0, 2: 0, 3: 0}
    f1_score_medians = []
    for i in range(4):
        f1_score_medians.append([list(rankings_medians_f1_score.keys())[i], median(allreps_f1[i])])
    f1_score_medians = sorted(f1_score_medians, key=itemgetter(1), reverse=True)
    
    s_bala_acc_all = '\n*All, rankings by balanced accuracy medians: '
    for i in range(len(bala_acc_medians)):
        s = '#' + str(i+1) + ': ' + fitnesses_name[bala_acc_medians[i][0]] + ' ' + str(bala_acc_medians[i][1]) + ', '
        s_bala_acc_all += s
        if bala_acc_medians[i][0] == 3:
            s_cm = '*' + s
        elif bala_acc_medians[i][0] == 2:
            s_reward = '*' + s
        rankings_medians_bala_acc[bala_acc_medians[i][0]] = i+1

    s_f1_all = '\n*All, rankings by f1 score medians: '
    for i in range(len(f1_score_medians)):
        s_f1 = '#' + str(i+1) + ': ' + fitnesses_name[f1_score_medians[i][0]] + ' ' + str(f1_score_medians[i][1]) + ', '
        s_f1_all += s_f1
        if f1_score_medians[i][0] == 3:
            s_ent_diagonal = '**' + s_f1
        elif f1_score_medians[i][0] == 2:
            s_ent = '**' + s_f1
        rankings_medians_f1_score[f1_score_medians[i][0]] = i+1

    rounds = 10000
    if bala_acc_medians[0][0] == names_to_indexes.index('Entropy With Diagonal'):
        pval = permutation_test(allreps_bala_acc[3], allreps_bala_acc[bala_acc_medians[1][0]], method='approximate', num_rounds=rounds,\
            func=lambda x, y: np.abs(np.median(x) - np.median(y)))
        pp = ' !' if pval < 0.05 else ''
        s_cm += 'pval: ' + '%.2E' % Decimal(pval) + pp + ', '
    if bala_acc_medians[1][0] == names_to_indexes.index('Entropy With Diagonal'):
        pval = permutation_test(allreps_bala_acc[3], allreps_bala_acc[bala_acc_medians[0][0]], method='approximate', num_rounds=rounds,\
            func=lambda x, y: np.abs(np.median(x) - np.median(y)))
        pp = ' =' if pval>=0.05 else ''
        s_cm += 'pval: ' + '%.2E' % Decimal(pval) + pp + ', '


    rounds = 10000
    if bala_acc_medians[0][0] == names_to_indexes.index('Entropy'):
        pval = permutation_test(allreps_bala_acc[2], allreps_bala_acc[bala_acc_medians[1][0]], method='approximate', num_rounds=rounds,\
            func=lambda x, y: np.abs(np.median(x) - np.median(y)))
        pp2 = ' !' if pval < 0.05 else ''
        s_reward += 'pval: ' + '%.2E' % Decimal(pval) + pp2 + ', '
    if bala_acc_medians[1][0] == names_to_indexes.index('Entropy'):
        pval = permutation_test(allreps_bala_acc[2], allreps_bala_acc[bala_acc_medians[0][0]], method='approximate', num_rounds=rounds,\
            func=lambda x, y: np.abs(np.median(x) - np.median(y)))
        pp2 = ' =' if pval>=0.05 else ''
        s_reward += 'pval: ' + '%.2E' % Decimal(pval) + pp2 + ', '

    # F1 SCORE !!!!
    rounds = 10000
    if f1_score_medians[0][0] == names_to_indexes.index('Entropy With Diagonal'):
        pval = permutation_test(allreps_f1[3], allreps_f1[f1_score_medians[1][0]], method='approximate', num_rounds=rounds,\
            func=lambda x, y: np.abs(np.median(x) - np.median(y)))
        pp3 = ' !' if pval < 0.05 else ''
        s_ent_diagonal += 'pval: ' + '%.2E' % Decimal(pval) + pp3 + ', '
    if f1_score_medians[1][0] == names_to_indexes.index('Entropy With Diagonal'):
        pval = permutation_test(allreps_f1[3], allreps_f1[f1_score_medians[0][0]], method='approximate', num_rounds=rounds,\
            func=lambda x, y: np.abs(np.median(x) - np.median(y)))
        pp3 = ' =' if pval>=0.05 else ''
        s_ent_diagonal += 'pval: ' + '%.2E' % Decimal(pval) + pp3 + ', '


    rounds = 10000
    if f1_score_medians[0][0] == names_to_indexes.index('Entropy'):
        pval = permutation_test(allreps_f1[2], allreps_f1[f1_score_medians[1][0]], method='approximate', num_rounds=rounds,\
            func=lambda x, y: np.abs(np.median(x) - np.median(y)))
        pp4 = ' !' if pval < 0.05 else ''
        s_ent += 'pval: ' + '%.2E' % Decimal(pval) + pp4 + ', '
    if f1_score_medians[1][0] == names_to_indexes.index('Entropy'):
        pval = permutation_test(allreps_f1[2], allreps_f1[f1_score_medians[0][0]], method='approximate', num_rounds=rounds,\
            func=lambda x, y: np.abs(np.median(x) - np.median(y)))
        pp4 = ' =' if pval>=0.05 else ''
        s_ent += 'pval: ' + '%.2E' % Decimal(pval) + pp4 + ', '

    



    fprint(res_file, '\n')
    fprint(res_file, "*Summary of experiment's results over " + str(n_replicates) + ' replicates: \n')
    fprint(res_file, s_bala_acc_all[:-2] + '\n')
    fprint(res_file, s_cm[:-2] + '\n')
    fprint(res_file, s_reward[:-2] + '\n')
    fprint(res_file, s_f1_all[:-2] + '\n')
    fprint(res_file, s_ent_diagonal[:-2] + '\n')
    fprint(res_file, s_ent[:-2] + '\n')

    s1, s2 = '*Fitnesses - balanced accuracy: ', '*Rankings (medians): '
    for i, rank in rankings_medians_bala_acc.items():
        s1 += fitnesses_name[i] + ', '
        s2 += str(rank) + ', '
    fprint(res_file, s1[:-2] + '\n')
    fprint(res_file, s2[:-2] + '\n')

    s3, s4 = '**Fitnesses - f1 score: ', '**Rankings (medians): '
    for i, rank in rankings_medians_f1_score.items():
        s3 += fitnesses_name[i] + ', '
        s4 += str(rank) + ', '
    fprint(res_file, s3[:-2] + '\n')
    fprint(res_file, s4[:-2] + '\n')



    s1, s2 = '*Times (algs), ', '*Times (all runs), '
    for i in range(4):
        s1 += fitnesses_name[list(rankings_medians_bala_acc.keys())[i]]+ ', '
        s2 += str(median(alltimes[i])) + ', '
    fprint(res_file, s1[:-2] + '\n')
    fprint(res_file, s2[:-2] + '\n')
    fprint(res_file, '\n')

##############
if __name__== "__main__":
  main()
