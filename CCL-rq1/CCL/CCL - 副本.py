import numpy as np
import pandas as pd
import copy
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import csv
import torch
import json
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



def evaluate(X_test, Y_test, model1, model2):

    X_test = Variable(torch.tensor(X_test).to(torch.float32))
    Y_test = Variable(torch.tensor(Y_test).to(torch.long))

    out1 = model1(X_test)
    out2 = model2(X_test)
    out = (out1 + out2) / 2

    result = out.argmax(dim=1)

    loc_all = sum(X_test[:, 4]) + sum(X_test[:, 5])
    loc_20 = loc_all * 0.2
    r_pre = np.zeros((len(result), 3))
    count_t = 0

    for il in range(len(result)):
        r_pre[il][0] = (X_test[il][4] + X_test[il][5])
        r_pre[il][1] = result[il]
        r_pre[il][2] = Y_test[il]
        if Y_test[il] == 1:
            count_t += 1
    idex = np.lexsort([r_pre[:, 0], -1 * r_pre[:, 1]])
    sorted_data = r_pre[idex, :]
    t_loc = 0
    il = 0
    count_effort = 0
    while t_loc < loc_20:
        t_loc += sorted_data[il][0]
        if sorted_data[il][2] == 1:
            count_effort = count_effort + 1
        il = il + 1
    effort = count_effort / count_t


    f1 = metrics.f1_score(Y_test, result, pos_label=1, average="binary")
    acc = accuracy_score(Y_test, result)
    prec = metrics.precision_score(Y_test, result, pos_label=1)  # 精确率
    recall = metrics.recall_score(Y_test, result, pos_label=1)  # 召回率
    fpr, tpr, thersholds = metrics.roc_curve(Y_test, result)
    roc_auc = metrics.auc(fpr, tpr)


    return acc, f1, prec, recall, roc_auc, effort


def train(train_loader, epoch, model1, optimizer1, model2, optimizer2):

    correct_all1 = 0
    correct_all2 = 0
    train_total1 = 0
    train_total2 = 0

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

        out_2 = model2(x_pruned)
        result2 = out_2.argmax(dim = 1)
        correct_2 = (result2.cpu() == s_pruned).sum()
        correct_all2 += correct_2
        train_total2 += x_pruned.size(0)

        criterion = nn.BCELoss(reduction = 'none')
        s_pruned = s_pruned.ravel()
        loss_1, loss_2, = loss_coteaching(out_1, out_2, s_pruned, rate_schedule[epoch], criterion)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

        f1_1 = metrics.f1_score(s_pruned, result1, pos_label=1, average="binary")
        f1_2 = metrics.f1_score(s_pruned, result2, pos_label=1, average="binary")

    train_acc1 = float(correct_all1) / float(train_total1)
    train_acc2 = float(correct_all2) / float(train_total2)

    return train_acc1, train_acc2

# warnings.filterwarnings('ignore')


csv_order = {0: 'jdt'}


def main():

    file_name = 'CCL.csv'
    f = open(file_name, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(
        ["datasets", "acc", "f1-score", "precise", "recall", "auc", "effort"])

    csv_string = csv_order[0]
    dataframe = pd.read_csv('oracle/' + csv_string + '.csv')
    v = dataframe.iloc[:1132, :]
    train_v = np.array(v)
    write_all = []
    write_all.append(csv_string)

    ob = train_v[:, 3:17]
    label = train_v[:, -2]
    ori_index = train_v[:, 0]
    label = label.reshape(-1, 1)
    # print(type(label))
    label = label.astype(int)

    loss_ranking_folders = {}


    seed = [94733, 16588, 1761, 59345, 27886, 80894, 22367, 65435, 96636, 89300]
    for ix in range(10):

        X_train = ob
        Y_train = label

        Y_train = Y_train.reshape((-1, 1))

        X_train = X_train.astype(np.float32)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        Y_train = Variable(torch.tensor(Y_train).to(torch.long))
        Y_train = Y_train.view(-1)

        epoch = 0

        model1 = MLP(numi, numh1, numh2, numo)
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)

        model1.train()
        for epoch in range(1, args.n_epoch):
            adjust_learning_rate(optimizer1, epoch)

            out_1 = model1(X_train)
            result1 = out_1.argmax(dim=1)

            loss = F.cross_entropy(out_1, Y_train, reduction='none')
            loss_1 = F.cross_entropy(out_1, Y_train)

            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()

            if (epoch + 1)==args.n_epoch:

                if ix == 0:
                    loss_ix1 = loss.detach().numpy()
                elif ix == 1:
                    loss_ix2 = loss.detach().numpy()
                    loss_all = np.add(loss_ix1, loss_ix2)
                else:
                    loss_all = np.add(loss_all, loss.detach().numpy())



                # loss_all = np.add(loss_all, loss.detach().numpy())
                #
                # data_with_losses = pd.DataFrame({'Index': range(len(loss)), 'Loss': loss.detach().numpy(), 'OriginalIndex': ori_index})
                # sorted_data = data_with_losses.sort_values(by='Loss', ascending=False)
                #
                # if ix + 7 == 10:
                #     top_data = sorted_data.head(100)
                #     loss_ranking_folders['times10'] = top_data

                # 输出排序后的结果
                # print(sorted_data)


        # print(csv_string + " is done~!")
    data_with_losses = pd.DataFrame({'Index': range(len(loss)), 'Loss': loss_all/10, 'OriginalIndex': ori_index})
    sorted_data = data_with_losses.sort_values(by='Loss', ascending=False)
    top_data = sorted_data.head(100)
    loss_ranking_folders['times10'] = top_data
    data_as_dict = {key: df.to_dict(orient='records') for key, df in loss_ranking_folders.items()}

    with open('output.json', 'w') as json_file:
        json.dump(data_as_dict, json_file)

    # json_content = loss_ranking_folders.to_json(orient='records')
    #
    # with open('output.json', 'w') as json_file:
    #     json_file.write(json_content)

    # with open("output.json", 'w') as json_file:
    #     json.dump(loss_ranking_folders, json_file)

if __name__ == '__main__':
    main()