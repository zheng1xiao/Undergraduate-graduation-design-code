import numpy as np
import pandas as pd
import copy
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import csv
import torch
from torch.autograd import Variable
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import argparse, sys
from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.pruning import get_noise_indices
from loader import GetLoader
from torch.utils.data import DataLoader
from modelMLP import MLP
from loss import loss_coteaching
import warnings
from cleanlab.latent_estimation import (
    compute_confident_joint,
    estimate_latent,
)


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.002)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.05)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='pairflip')
parser.add_argument('--num_gradual', type = int, default = 50, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=5)
# parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_workers', type=int, default=0, help='how many subprocesses to use for data loading')    # 多线程导致代码重复执行
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=120)

args = parser.parse_args()
batch_size = 128
numi = 14 # 输入层节点数
numh1 = 30  # 隐含层节点数
numh2 = 30
numo = 2  # 输出层节点数

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)    # 设置当前GPU的随机生成数种子

learning_rate = args.lr

if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

warnings.filterwarnings("ignore")
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1

# define drop rate schedule
rate_schedule = np.ones(args.n_epoch)*forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)    # 每个epoch去掉的样本比例, exponent=1时适用

def evaluate(X_test, Y_test, model1):

    X_test = Variable(torch.tensor(X_test).to(torch.float32))
    Y_test = Variable(torch.tensor(Y_test).to(torch.long))

    out1 = model1(X_test)
    result1 = out1.argmax(dim=1)

    loc_all = sum(X_test[:, 4]) + sum(X_test[:, 5])
    loc_20 = loc_all * 0.2
    r_pre1 = np.zeros((len(result1), 3))
    count_t1 = 0

    for il in range(len(result1)):
        r_pre1[il][0] = (X_test[il][4] + X_test[il][5])
        r_pre1[il][1] = result1[il]
        r_pre1[il][2] = Y_test[il]
        if Y_test[il] == 1:
            count_t1 += 1
    idex = np.lexsort([r_pre1[:, 0], -1 * r_pre1[:, 1]])
    sorted_data = r_pre1[idex, :]
    t_loc = 0
    il = 0
    count_effort1 = 0
    while t_loc < loc_20:
        t_loc += sorted_data[il][0]
        if sorted_data[il][2] == 1:
            count_effort1 = count_effort1 + 1
        il = il + 1
    effort1 = count_effort1 / count_t1

    f1_1 = metrics.f1_score(Y_test, result1, pos_label=1, average="binary")
    acc_1 = accuracy_score(Y_test, result1)
    prec1 = metrics.precision_score(Y_test, result1, pos_label=1)  # 精确率
    recall1 = metrics.recall_score(Y_test, result1, pos_label=1)  # 召回率
    fpr1, tpr1, thersholds1 = metrics.roc_curve(Y_test, result1)
    roc_auc1 = metrics.auc(fpr1, tpr1)

    return acc_1, f1_1, prec1, recall1, roc_auc1, effort1


def train(train_loader, epoch, model1, optimizer1):

    correct_all1 = 0
    train_total1 = 0

    for i, (x_data, labels, indexes) in enumerate(train_loader):
        if i>args.num_iter_per_epoch:
            break

        x_pruned = Variable(torch.tensor(x_data).to(torch.float32))
        s_pruned = Variable(torch.tensor(labels).to(torch.long))

        out_1 = model1(x_pruned)
        result1 = out_1.argmax(dim = 1)
        correct_1 = (result1.cpu() == s_pruned).sum()
        correct_all1 += correct_1
        train_total1 += x_pruned.size(0)

        s_pruned = s_pruned.ravel()
        loss_sum = F.cross_entropy(out_1, s_pruned, reduction='none')
        loss_1 = torch.sum(loss_sum) / len(loss_sum)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()

        prec1 = accuracy_score(s_pruned, result1)
        f1_1 = metrics.f1_score(s_pruned, result1, pos_label=1, average="binary")

    train_acc1 = float(correct_all1) / float(train_total1)

    return train_acc1

# warnings.filterwarnings('ignore')


csv_order = {0: 'activemq', 1: 'camel', 2: 'derby', 3: 'geronimo', 4: 'hbase', 5: 'hcommon', 6: 'mahout', 7: 'openjpa',
             8: 'pig', 9: 'tuscany'}
csv_num = {0: 1245, 1: 2018, 2: 1153, 3: 1856, 4: 1681, 5: 2670, 6: 420, 7: 692, 8: 467, 9: 1506}


def main():

    file_name = 'c1-newB.csv'
    f = open(file_name, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(
        ["datasets", "acc", "f1-score", "precise", "recall", "auc", "effort"])

    for i in range(10):
        csv_string = csv_order[i]
        dataframe = pd.read_csv('dataset/' + csv_string + '.csv')
        v = dataframe.iloc[:]
        train_v = np.array(v)
        write_all = []
        write_all.append(csv_string)

        ob = train_v[:, 0:14]
        label = train_v[:, -4]
        test_label = train_v[:, -1]
        label = label.reshape(-1, 1)

        acc1_all = []
        f11_all = []
        prec1_all = []
        recall1_all = []
        auc1_all = []
        effort1_all = []

        seed = [94733, 16588, 1761, 59345, 27886, 80894, 22367, 65435, 96636, 89300]
        for ix in range(10):
            sfolder = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed[ix])

            acc1 = []
            f11 = []
            pre1 = []
            rec1 = []
            auc_1 = []
            teffort1 = []

            for train_index, test_index in sfolder.split(ob, label):

                X_train, X_test = ob[train_index], ob[test_index]
                Y_train, Y_test = label[train_index], test_label[test_index]

                Y_train = Y_train.reshape((-1, 1))
                train_data = GetLoader(X_train, Y_train)
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

                epoch = 0
                train_acc1 = 0

                model1 = MLP(numi, numh1, numh2, numo)
                optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)

                model1.train()

                for epoch in range(1, args.n_epoch):
                    adjust_learning_rate(optimizer1, epoch)
                    train_acc1 = train(train_loader, epoch, model1, optimizer1)

                model1.eval()

                test_acc, test_f1, prec, recall, roc, effort = evaluate(X_test, Y_test, model1)
                print('%s Epoch [%d/%d] Test Accuracy on the %s test: Model %.4f, f1-score: %.4f, auc: %.4f, effort-aware: %.4f ' % (
                   csv_string, epoch + 1, args.n_epoch, len(Y_test), test_acc, test_f1, roc, effort))

                acc1.append(test_acc)
                f11.append(test_f1)
                pre1.append(prec)
                rec1.append(recall)
                auc_1.append(roc)
                teffort1.append(np.mean(effort))

            acc1_all.append(np.mean(acc1))
            f11_all.append(np.mean(f11))
            prec1_all.append(np.mean(pre1))
            recall1_all.append(np.mean(rec1))
            auc1_all.append(np.mean(auc_1))
            effort1_all.append(np.mean(teffort1))

        write_all.append(np.mean(acc1_all))
        write_all.append(np.mean(f11_all))
        write_all.append(np.mean(prec1_all))
        write_all.append(np.mean(recall1_all))
        write_all.append(np.mean(auc1_all))
        write_all.append(np.mean(effort1_all))
        print(np.mean(acc1_all), np.mean(mcc1_all), np.mean(f11_all), np.mean(prec1_all), np.mean(recall1_all), np.mean(auc1_all), np.mean(effort1_all))

        csv_writer.writerow(write_all)
        print(csv_string + " is done~!")

if __name__ == '__main__':
    main()