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
    file_name = 'rsds-crossproject.csv'
    f = open(file_name, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["datasets", "f1-score", "precise", "recall", "auc", "effort", "mcc"])

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

        train_data = np.append(X_train, y_train, axis=1)

        new_train_data = RSDS.RSDS_fun(train_data)

        x_pruned = new_train_data[:, :-1]
        y_pruned = new_train_data[:,-1]
        s_pruned = y_pruned.reshape(-1, 1)

        log_reg1 = RandomForestClassifier()
        log_reg1.fit(x_pruned, s_pruned)
        pre1 = log_reg1.predict(X_test)
        y_test_1 = y_test.ravel()
        y_original = metrics.f1_score(y_test_1, pre1, pos_label=1, average="binary")
        fpr, tpr, thersholds = metrics.roc_curve(y_test_1, pre1)
        roc_auc = metrics.auc(fpr, tpr)
        prec = metrics.precision_score(y_test_1, pre1, pos_label=1)  # 精确率
        recall = metrics.recall_score(y_test_1, pre1, pos_label=1)  # 召回率
        mcc = matthews_corrcoef(y_test_1, pre1)

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
        effort = count_effort / count_t1

        write_all.append(y_original)
        write_all.append(roc_auc)
        write_all.append(prec)
        write_all.append(recall)
        write_all.append(mcc)
        write_all.append(effort)
        csv_writer.writerow(write_all)
        print(y_original,roc_auc)
        print(csv_string_test+ " is done~!")

if __name__ == '__main__':
    con_learn()