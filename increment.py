import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.nn import functional as F

torch.backends.cudnn.enabled = False
cuda_available = False
if torch.cuda.is_available():
    cuda_available = True

def compute_offsets(task, nc_per_task):
    offset1 = task * nc_per_task
    offset2 = (task + 1) * nc_per_task
    return offset1, offset2

def to_one_hot(labels: torch.Tensor, num_class: int):
    y = torch.zeros(labels.shape[0], num_class)
    for i, label in enumerate(labels):
        y[i, label] = 1
    return y

class increLearning(torch.nn.Module):
    def __init__(self, net, tasks):
        super(increLearning, self).__init__()
        self.model = net  # 预测
        self.tasks = tasks
        self.criterion = torch.nn.CrossEntropyLoss()
        self.old_model = None
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.nc_per_task = 1  # 每个任务的输出个数
        self.n_outputs = 6
        self.ep_mem = torch.FloatTensor(self.tasks * 20, 1, 200)
        self.ep_labels = torch.LongTensor(self.tasks * 20)
        # 为过去的任务制作梯度矩阵梯度
        self.R = torch.zeros((self.tasks, self.tasks))
        if cuda_available:
            self.R = self.R.cuda()

    def forward(self, data, t):
        pre = self.model(data)
        # 确保我们预测当前任务中的类别
        offset1 = int(t * self.nc_per_task)
        offset2 = int((t + 1) * self.nc_per_task)
        if offset1 > 0:
            pre[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_outputs:
            pre[:, offset2:self.n_outputs].data.fill_(-10e10)
        return pre

    def forward2(self, out, t):
        # 确保我们预测当前任务中的类别
        offset2 = int(t * self.nc_per_task)
        if offset2 < self.n_outputs:
            out[:, offset2:self.n_outputs].data.fill_(-10e10)
        return out

    def train(self, data_loader, task, epoch):
        self.model.train()
        self.cur_task = task
        running_loss = 0.0
        for data_train, label_train in data_loader:
            data_train = data_train.type(torch.FloatTensor)
            label_train = label_train.type(torch.LongTensor)
            # if cuda_available:
            #     data_train = data_train.cuda()
            #     label_train = label_train.cuda()

            self.optim.zero_grad()
            offset1, offset2 = compute_offsets(task, self.nc_per_task)
            pred = self.forward(data_train,task)
            pred = pred[:, offset1: offset2]
            loss_pre = self.criterion(pred, label_train - offset1) #交叉损失熵
            if self.old_model == None:
                loss = loss_pre
            else:
                data_ = self.ep_mem[: self.cur_task*20, :]
                label_ = self.ep_labels[:self.cur_task*20]
                output_old = self.old_model(data_)
                output_old = self.forward2(output_old,task)[:, : offset1]
                lab_dist = Variable(output_old, requires_grad=False)
                output_new = self.model(data_)
                output_new = self.forward2(output_new, task)[:, : offset1]

                output_new_soft = torch.softmax(output_new / 2, dim=1)
                output_old_soft = torch.softmax(lab_dist / 2, dim=1)
                loss_dist = F.binary_cross_entropy(output_new_soft,output_old_soft)
                loss_soft = self.criterion(output_new,label_)
                loss = loss_pre + loss_dist + loss_soft
            loss.backward()
            running_loss += loss.item()

            self.optim.step()

        msg = '[%d / %d] AVG. loss: %.3f\n' % (task + 1, epoch,running_loss)
        print(msg)

    def eval(self, data_loader, task, task_num):
        total = 0
        correct = 0
        self.model.eval()
        for data_test, label_test in data_loader:
            label_test = label_test.type(torch.LongTensor)
            data_test = data_test.type(torch.FloatTensor)
            # if cuda_available:
            #     data_test = data_test.cuda()
            #     label_test = label_test.cuda()
            with torch.no_grad():
                pre = self.forward(data_test, task)
            _, predicted = torch.max(pre, dim=1)
            total += label_test.size(0)
            correct += (predicted == label_test).sum().item()
        accuracy = correct / total
        print('Testing Accuracy:{:.3f}'.format(accuracy))
        self.R[self.cur_task][task] = 100 * accuracy

        if task == task_num:
            filename = './model/increment_%d_net.pkl' % (task)
            print(filename)
            torch.save(self.model, filename)
            self.old_model = torch.load(filename)
            # self.old_model.to(device)
            self.old_model.eval()

    def after(self,cur_task, memory, l):
        self.ep_mem[20*cur_task:20*(cur_task+1), :] = memory
        self.ep_labels[20*cur_task:20*(cur_task+1)] = torch.from_numpy(l.squeeze())