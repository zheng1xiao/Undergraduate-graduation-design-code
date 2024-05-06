import numpy as np
import pandas as pd
import copy
import random
import xlwt
import csv
import RSDS
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import IsolationForest
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings


#baseline1:RSDS

warnings.filterwarnings('ignore')

csv_order = {0: 'activemq', 1: 'camel', 2: 'derby', 3: 'geronimo', 4: 'hbase', 5: 'hcommon', 6: 'mahout', 7: 'openjpa',
             8: 'pig', 9: 'tuscany'}
csv_num = {0: 1245, 1: 2018, 2: 1153, 3: 1856, 4: 1681, 5: 2670, 6: 420, 7: 692, 8: 467, 9: 1506}


def con_learn():
    file_name = 'rsds-198.csv'
    f = open(file_name, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["datasets", "f1-ori", "f1-rsds", "auc-ori", "auc-rsds", "prec-ori", "prec-rsds", "recall-ori", "recall-rsds",
                         "mcc-ori", "mcc-rsds", "effort-ori", "effort-rsds"])

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
            y_or = []
            y_or1 = []

            auc_or = []  # 原始的auc值
            auc_or1 = []  # RSDS的auc值

            prec_or = []
            prec_or1 = []

            recall_or = []
            recall_or1 = []

            mcc_or = []  # 原始mcc值
            mcc_or1 = []  # RSDS mcc值

            effort_or = []  # 原始Recall@20%值
            effort_or1 = []  # RSDS的Recall@20%值
            for train_index, test_index in sfolder.split(ob, label):
                X_train, X_test = ob[train_index], ob[test_index]
                y_train, y_test = label[train_index], label[test_index]

                train_data = np.append(X_train, y_train, axis=1)

                new_train_data = RSDS.RSDS_fun(train_data)

                x_pruned = new_train_data[:, :-1]
                y_pruned = new_train_data[:,-1]
                s_pruned = y_pruned.reshape(-1, 1)

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

                log_reg1 = RandomForestClassifier(random_state=seed[ix])
                log_reg1.fit(x_pruned, s_pruned)
                pre1 = log_reg1.predict(X_test)
                y_test_1 = y_test.ravel()
                y_original = metrics.f1_score(y_test_1, pre1, pos_label=1, average="binary")
                y_or1.append(y_original)
                fpr, tpr, thersholds = metrics.roc_curve(y_test_1, pre1)
                roc_auc = metrics.auc(fpr, tpr)
                auc_or1.append(roc_auc)
                prec = metrics.precision_score(y_test_1, pre1, pos_label=1)  # 精确率
                prec_or1.append(prec)
                recall = metrics.recall_score(y_test_1, pre1, pos_label=1)  # 召回率
                recall_or1.append(recall)
                mcc = matthews_corrcoef(y_test_1, pre1)
                mcc_or1.append(mcc)

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
                effort_or1.append(count_effort / count_t1)

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
    con_learn()