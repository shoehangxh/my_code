import torch
import torch.nn as nn
import torchvision.models as m
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
import copy
import time
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


vgg16_bn = m.vgg16_bn()
resnet18_ = m.resnet18()

class net(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
                                   , nn.BatchNorm2d(64)
                                   , nn.ReLU(inplace=True))
        self.block2 = vgg16_bn.features[7:14]
        self.block3 = resnet18_.layer3
        self.avgpool = resnet18_.avgpool
        self.fc = nn.Linear(in_features=256, out_features=10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block3(self.block2(x))
        x = self.avgpool(x)
        x = x.view(x.shape[0], 256)
        x = self.fc(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self,in_,out_=10,**kwargs):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_,out_,**kwargs)
                                  ,nn.BatchNorm2d(out_)
                                  ,nn.ReLU(inplace=True)
                                 )
    def forward(self,x):
        x = self.conv(x)
        return x

class MyNet2(nn.Module):
    def __init__(self, in_channels=1, out_features=10):
        super().__init__()
        self.block1 = nn.Sequential(BasicConv2d(in_=in_channels, out_=32, kernel_size=5, padding=2)
                                    , BasicConv2d(32, 32, kernel_size=5, padding=2)
                                    , nn.MaxPool2d(2)
                                    , nn.Dropout2d(0.25))
        self.block2 = nn.Sequential(BasicConv2d(32, 64, kernel_size=3, padding=1)
                                    , BasicConv2d(64, 64, kernel_size=3, padding=1)
                                    , BasicConv2d(64, 64, kernel_size=3, padding=1)
                                    , nn.MaxPool2d(2)
                                    , nn.Dropout2d(0.25))

        self.classifier_ = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256)
            , nn.BatchNorm1d(256)  # 此时数据已是二维，因此需要BatchNorm1d
            , nn.ReLU(inplace=True)
            , nn.Linear(256, out_features)
            , nn.LogSoftmax(1)
        )

    def forward(self, x):
        x = self.block2(self.block1(x))
        x = x.view(-1, 64 * 7 * 7)
        output = self.classifier_(x)
        return output


def train_model(model, traindataloader, train_rate, criterion, optimizer, num_epochs=25):
    # model:网络模型
    # trainloader:训练数据集，会切分为训练集和验证集
    # train_rate:训练集batchsize百分比
    # criterion:损失函数
    # optimizer:优化方法
    # num_epochs:训练的轮数
    ##计算训练使用的batch数量
    batch_num = len(traindataloader)
    train_batch_num = round(batch_num * train_rate)
    ##复制模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        ##每个epoch有两个训练阶段
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        for step, (b_x, b_y) in enumerate(traindataloader):
            if step < train_batch_num:
                model.train()  ##设置模型为训练模式
                output = model(b_x)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * b_x.size(0)
                train_corrects += torch.sum(pre_lab == b_y.data)
                train_num += b_x.size(0)
            else:
                model.eval()  # 3设置模型为评估模式
                output = model(b_x)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, b_y)
                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)

        ##计算一个epoch在训练集和验证集上的损失和精度
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        print('Train Loss:{:.4f}  Train Acc: {:.4f}'.format(train_loss_all[-1], train_acc_all[-1]))
        print('Val Loss:{:.4f}  Val Acc:{:.4f}'.format(val_loss_all[-1], val_acc_all[-1]))
        print('/t')
        ##拷贝模型最高精度下的参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time() - since
        print("Train and val complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))
    ##使用最好模型的参数
    model.load_state_dict(best_model_wts)
    train_process = pd.DataFrame(
        data={
            "epoch": range(num_epochs),
            "train_loss_all": train_loss_all,
            "val_loss_all": val_loss_all,
            "train_acc_all": train_acc_all,
            "val_acc_all": val_acc_all
        }
    )
    return model, train_process
batch_size = 64
fmnist_train = torchvision.datasets.FashionMNIST(root='D:\datasets'
                                           , train=True #根据类的不同，参数可能发生变化
                                           , download=False #未下载则设置为True
                                           , transform=transforms.ToTensor()
                                          )
fmnist_test = torchvision.datasets.FashionMNIST(root='D:\datasets'
                                           , train=False #根据类的不同，参数可能发生变化
                                           , download=False #未下载则设置为True
                                           , transform=transforms.ToTensor()
                                          )
train = data.DataLoader(fmnist_train, batch_size, shuffle=True, num_workers=2)
test_data_x = fmnist_test.data.type(torch.FloatTensor)/255.0
test_data_x = torch.unsqueeze(test_data_x, dim=1)
test_data_y = fmnist_test.targets
model = MyNet2()
traindataloader = train
train_rate = 0.8
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
criterion = nn.CrossEntropyLoss()
if __name__ == '__main__':
    myconvnet, train_process = train_model(model, traindataloader, train_rate, criterion, optimizer, num_epochs=1)
    plt.figure(figsize=(12, 4))
    ##损失函数
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process.epoch, train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    ##精度
    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process.epoch, train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()
    ##对测试集进行预测，并可视化预测结果
    myconvnet.eval()
    output = myconvnet(test_data_x)
    pre_lab = torch.argmax(output, 1)
    acc = accuracy_score(test_data_y, pre_lab)
    print("在测试集上的预测精度为：", acc)

    ##计算混淆矩阵并可视化
    conf_mat = confusion_matrix(test_data_y, pre_lab)
    class_label = fmnist_train.classes
    df_cm = pd.DataFrame(conf_mat, index=class_label, columns=class_label)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()