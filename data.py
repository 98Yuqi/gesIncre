import scipy.io as scio
import numpy as np
import os,sys
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset,DataLoader

def load_data(path):
    list = os.listdir((path))
    data = []
    label = []

    for file in list:
        data_path = os.path.join(path, file)
        data_1 = scio.loadmat(data_path)['csi']
        data_1 = np.expand_dims(data_1, axis=0)
        label_1 = int(file.split('-')[1])

        data.append(data_1.tolist())
        label.append(label_1)

    label_total = np.array(label)
    data_total = np.array(data).squeeze(axis=3)

    return data_total, label_total

def getDataLoader(data_total, label_total, classes):
    train_loader = {}
    test_loader = {}
    for i in range(6):
        datas, labels = [], []
        for label in range(classes*i, classes*(i+1)):
            data = data_total[np.array(label_total) == label]  # 找到指定标签的数据
            datas.append(data)  # 将数据拼接起来
            labels.append(np.full((data.shape[0]), label))  # 将标签拼接起来`
        Datas, Labels = concatenate(datas, labels)
        [traindata, testdata, trainlabels, testlabels] = train_test_split(Datas, Labels,test_size=0.3)
        print('\nTrain on ' + str(trainlabels.shape[0]) + ' samples\n' + \
                       'Test on ' + str(testlabels.shape[0]) + ' samples\n')

        train_dataset = TensorDataset(torch.tensor(traindata), torch.tensor(trainlabels))
        test_dataset = TensorDataset(torch.tensor(testdata), torch.tensor(testlabels))
        test_loader[i] = DataLoader(test_dataset, batch_size=16, shuffle=True)
        train_loader[i] = DataLoader(train_dataset, batch_size=16, shuffle=True)

    return train_loader,test_loader


def concatenate(datas,labels):
    con_data=datas[0]
    con_label=labels[0]
    for i in range(1,len(datas)):
        con_data=np.concatenate((con_data,datas[i]),axis=0)
        con_label=np.concatenate((con_label,labels[i]),axis=0)
    return con_data,con_label