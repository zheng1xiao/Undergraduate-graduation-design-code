import numpy as np
import pandas as pd
import copy
import random
import xlwt
import csv
from sklearn.model_selection import KFold, StratifiedKFold
from collections import Counter
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import matthews_corrcoef
from cleanlab.classification import LearningWithNoisyLabels
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score
from cleanlab.pruning import get_noise_indices
import warnings
from cleanlab.latent_estimation import (
    compute_confident_joint,
    estimate_latent,
)


def get_noise(label_new, pre_new, xall_new, X_test, y_test):
    label_new1=copy.deepcopy(label_new)
    pre_new1 = copy.deepcopy(pre_new)
    xall_new1 = copy.deepcopy(xall_new)
    XX = copy.deepcopy(X_test)
    y_test_no = copy.deepcopy(y_test)

    ordered_label_errors=[]
    count_no=0
    for ii in range(len(label_new1)):
        r_pre1 = np.zeros((len(label_new1), 2))
        for jj in range(len(label_new1)):
            for zz in range(14):
                r_pre1[jj][0]=r_pre1[jj][0]+(xall_new1[ii][zz]-xall_new1[jj][zz])**2
            r_pre1[jj][0] = r_pre1[jj][0]**0.5
            r_pre1[jj][1] = label_new1[jj]
        idex = np.lexsort([r_pre1[:, 0]])
        sorted_data = r_pre1[idex, :]
        countc = 0
        for jj in range(6):
            if(label_new1[ii]!=sorted_data[jj][1]):
                countc = countc+1
        the_1 = (countc-1)/5
        if the_1>=0.6:
            ordered_label_errors.append(False)
            count_no=count_no+1
        else:
            ordered_label_errors.append(True)
        if count_no>len(label_new1)*0.01:
            break;
    for ii in range(len(label_new1)-len(ordered_label_errors)):
        ordered_label_errors.append(True)

    #x_mask = ~ordered_label_errors
    x_pruned = xall_new1[ordered_label_errors]
    s_pruned = label_new1[ordered_label_errors]

    log_reg = RandomForestClassifier()
    # log_reg1 = LogisticRegression(solver='liblinear')
    log_reg.fit(x_pruned, s_pruned)
    pre1 = log_reg.predict(XX)
    y_test_11 = y_test_no.ravel()
    y_original = metrics.f1_score(y_test_11, pre1, pos_label=1, average="binary")

    fpr, tpr, thersholds = metrics.roc_curve(y_test_11, pre1)
    roc_auc = metrics.auc(fpr, tpr)
    prec = metrics.precision_score(y_test_11, pre1, pos_label=1)  # 精确率
    recall = metrics.recall_score(y_test_11, pre1, pos_label=1)  # 召回率
    mcc = matthews_corrcoef(y_test_11, pre1)

    loc_all = sum(XX[:, 4]) + sum(XX[:, 5])
    loc_20 = loc_all * 0.2
    r_pre1 = np.zeros((len(pre1), 3))
    count_t1 = 0
    for il in range(len(pre1)):
        r_pre1[il][0] = (XX[il][4] + XX[il][5])
        r_pre1[il][1] = pre1[il]
        r_pre1[il][2] = y_test[il]
        if y_test[il] == 1:
            count_t1 += 1
    idex = np.lexsort([r_pre1[:, 0], -1 * r_pre1[:, 1]])
    sorted_data = r_pre1[idex, :]
    t_loc = 0
    il = 0
    count_effort = 0
    while t_loc < loc_20:
        t_loc += sorted_data[il][0]
        if sorted_data[il][2] == 1:
            count_effort = count_effort + 1
        il = il + 1
    eff=count_effort / count_t1
    return y_original, roc_auc, prec, recall, mcc, eff


warnings.filterwarnings('ignore')

csv_order = {0: 'activemq', 1: 'camel', 2: 'derby', 3: 'geronimo', 4: 'hbase', 5: 'hcommon', 6: 'mahout', 7: 'openjpa',
             8: 'pig', 9: 'tuscany'}
csv_num = {0: 1245, 1: 2018, 2: 1153, 3: 1856, 4: 1681, 5: 2670, 6: 420, 7: 692, 8: 467, 9: 1506}


def main():
    file_name = 'CLNI-crossproject.csv'
    f = open(file_name, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["datasets", "f1", "auc", "precise", "recall", "mcc", "effort"])

    for i in range(10):
        data_num = 100
        csv_string_test = csv_order[i]
        dataframe_test = pd.read_csv('dataset/' + csv_string_test + '.csv')
        dt = dataframe_test.iloc[:]
        dt = np.array(dt)
        ob = dt[:data_num, 0:14]

        label = dt[:data_num, -1]
        label = label.reshape(-1, 1)

        X_test = ob
        y_test = label

        X_train = []
        Y_train = []

        for j in range(10):
            if j != i:
                csv_string_train = csv_order[j]
                dataframe_train = pd.read_csv('dataset/' + csv_string_train + '.csv')
                dtrain = dataframe_train.iloc[:]
                dtrain = np.array(dtrain)
                ob_train = dtrain[:data_num, 0:14]

                label_train = dtrain[:data_num, -1]
                X_train.extend(ob_train)
                Y_train.extend(label_train)

        X_train = np.array(X_train)
        y_train = np.array(Y_train).reshape(-1,1)

        write_all = []
        write_all.append(csv_string_test)

        psx1 = np.zeros((len(y_train), 2))

        y_original, thresholds, prec, recall, mcc, eff = get_noise(y_train, psx1, X_train, X_test, y_test)

        write_all.append(y_original)
        write_all.append(thresholds)
        write_all.append(prec)
        write_all.append(recall)
        write_all.append(mcc)
        write_all.append(eff)
        csv_writer.writerow(write_all)
        print(y_original,thresholds)
        print(csv_string_test + " is done~!")


if __name__ == '__main__':
    main()