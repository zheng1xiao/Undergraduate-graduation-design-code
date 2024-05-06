import torch
import numpy as np
from torch.utils.data import DataLoader

# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels, index
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    # 随机生成数据，大小为10 * 20列
    source_data = np.random.rand(10, 20)
    # 随机生成标签，大小为10 * 1列
    source_label = np.random.randint(0, 2, (10, 1))
    # 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
    torch_data = GetLoader(source_data, source_label)
    print(torch_data)
    # 读取数据
    datas = DataLoader(torch_data, batch_size=6, shuffle=True, drop_last=False, num_workers=0)
    # for i, data in enumerate(datas):
    #     # i表示第几个batch， data表示该batch对应的数据，包含data和对应的labels
    #     print("第 {} 个Batch \n{}".format(i, data))
