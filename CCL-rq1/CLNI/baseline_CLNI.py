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


def get_noise(label_new, pre_new, xall_new, X_test, y_test, seed_t):
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

    log_reg = RandomForestClassifier(random_state=seed_t)
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
    file_name = 'CLNI.csv'
    f = open(file_name, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["datasets", "f1-ori", "f1-clni", "auc-ori", "auc-clni",  "prec-cl", "prec-clni","recall-ori",
                         "recall-clni", "mcc-ori", "mcc-clni","effort-ori", "effort-clni"])

    for i in range(10):
        csv_string = csv_order[i]
        dataframe = pd.read_csv('dataset/' + csv_string + '.csv')
        v = dataframe.iloc[:]
        train_v = np.array(v)
        ori_all = []
        ori_all.append(csv_string)

        ob = train_v[:, 0:14]
        label = train_v[:, -1]
        label = label.reshape(-1, 1)

        yor_all = []
        yor_all1 = []

        auc_all = []
        auc_all1 = []

        prec_all = []
        prec_all1 = []

        recall_all = []
        recall_all1 = []

        mcc_all = []
        mcc_all1 = []

        effort_all = []
        effort_all1 = []

        seed=[94733,16588,1761,59345,27886,80894,22367,65435,96636,89300]
        for ix in range(10):
            sfolder = KFold(n_splits=10, shuffle=True,random_state= seed[ix])
            y_or = []     #原始f值
            y_or1 = []     #CLNI f值

            auc_or = []  # 原始auc值
            auc_or1 = []  # CLNI auc值

            prec_or= []
            prec_or1 = []

            recall_or = []
            recall_or1 = []

            mcc_or = []  # 原始mcc值
            mcc_or1 = []  # CLNI mcc值

            effort_or = []  # 原始Recall@20%值
            effort_or1 = []  # CLNI Recall@20%值
            for train_index, test_index in sfolder.split(ob, label):
                X_train, X_test = ob[train_index], ob[test_index]
                y_train, y_test = label[train_index], label[test_index]

                psx1 = np.zeros((len(y_train), 2))

                log_reg = RandomForestClassifier(random_state=seed[ix])
                log_reg.fit(X_train, y_train)
                pre1 = log_reg.predict(X_test)
                y_test_1 = y_test.ravel()
                y_original = metrics.f1_score(y_test_1, pre1, pos_label=1, average="binary")
                y_or.append(y_original)
                fpr, tpr, thersholds = metrics.roc_curve(y_test_1, pre1)
                roc_auc = metrics.auc(fpr, tpr)
                auc_or.append(roc_auc)
                prec = metrics.precision_score(y_test_1, pre1, pos_label=1)  # 精确率
                prec_or.append(prec)
                recall = metrics.recall_score(y_test_1, pre1, pos_label=1)  # 召回率
                recall_or.append(recall)
                mcc = matthews_corrcoef(y_test_1, pre1)
                mcc_or.append(mcc)

                loc_all = sum(X_test[:, 4]) + sum(X_test[:, 5])
                loc_20 = loc_all * 0.2
                r_pre1 = np.zeros((len(pre1), 3))
                count_t1 = 0
                for il in range(len(pre1)):
                    r_pre1[il][0] = (X_test[il][4] + X_test[il][5])
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
                effort_or.append(count_effort / count_t1)

                y_original, thresholds2, prec2, recall2, mcc2, eff2 = get_noise(y_train, psx1, X_train, X_test, y_test, seed[ix])
                y_or1.append(y_original)
                auc_or1.append(thresholds2)
                prec_or1.append(prec2)
                recall_or1.append(recall2)
                mcc_or1.append(mcc2)
                effort_or1.append(eff2)

            yor_all.append(np.mean(y_or))
            yor_all1.append(np.mean(y_or1))
            auc_all.append(np.mean(auc_or))
            auc_all1.append(np.mean(auc_or1))
            prec_all.append(np.mean(prec_or))
            prec_all1.append(np.mean(prec_or1))
            recall_all.append(np.mean(recall_or))
            recall_all1.append(np.mean(recall_or1))
            mcc_all.append(np.mean(mcc_or))
            mcc_all1.append(np.mean(mcc_or1))
            effort_all.append(np.mean(effort_or))
            effort_all1.append(np.mean(effort_or1))

        ori_all.append(np.mean(yor_all))
        ori_all.append(np.mean(yor_all1))
        ori_all.append(np.mean(auc_all))
        ori_all.append(np.mean(auc_all1))
        ori_all.append(np.mean(prec_all))
        ori_all.append(np.mean(prec_all1))
        ori_all.append(np.mean(recall_all))
        ori_all.append(np.mean(recall_all1))
        ori_all.append(np.mean(mcc_all))
        ori_all.append(np.mean(mcc_all1))
        ori_all.append(np.mean(effort_all))
        ori_all.append(np.mean(effort_all1))
        csv_writer.writerow(ori_all)
        print(csv_string + " is done~!")


if __name__ == '__main__':
    main()