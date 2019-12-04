# -*- coding: utf-8 -*-
import torch
import Net
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from torch.nn import DataParallel
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

plt.switch_backend('agg')
import pickle
import time
import os
from tqdm import tqdm

logDir = './log_raw'
imgDir = 'Raw'

from tensorboardX import SummaryWriter

# if os.path.exists(logDir):
#     pass
# else:
#     os.mkdir(logDir)
# writer = SummaryWriter(logDir)
os.makedirs(imgDir,exist_ok=True)
batch_size = 500
learning_rate = 1e-2
# num_epoches = 20
dataTransform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])
     ]
)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
train_dataset = datasets.MNIST(
    root="/home/ubuntu/LRP/FR-Loss-on-Mnist-master/MNIST", train=True, transform=dataTransform, download=False
)
test_dataset = datasets.MNIST(
    root="/home/ubuntu/LRP/FR-Loss-on-Mnist-master/MNIST", train=False, transform=dataTransform, download=False
)
# print(len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

activation = {}

EbeddingNet = Net.EModel()
EbeddingNet.to(device)
EbeddingNet = nn.DataParallel(EbeddingNet, device_ids=range(torch.cuda.device_count()))
# EbeddingNet.cuda()
head = Net.Linear()
head.to(device)
# head = nn.DataParallel(head, device_ids=range(torch.cuda.device_count()))
# head.cuda()
myCrossEntropyLoss = nn.CrossEntropyLoss()

def visualize(feat, labels, epoch):
    colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
              '#ff00ff', '#990000', '#999900', '#009900', '#009999']

    fig = Figure(figsize=(6, 6), dpi=100)
    fig.clf()
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    feat = feat.data.cpu().numpy()
    labels = labels.data.cpu().numpy()

    for i in range(10):
        ax.scatter(feat[labels == i, 0], feat[labels == i, 1], c=colors[i], s=1)
        ax.text(feat[labels == i, 0].mean(), feat[labels == i, 1].mean(), str(i), color='black', fontsize=12)
    ax.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    ax.text(0, 0, "epoch=%d" % epoch)
    canvas.draw()

    if (os.path.exists(imgDir)):
        pass
    else:
        os.makedirs(imgDir)
    fig.savefig(imgDir + '/epoch=%d.jpg' % epoch)
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    tt = transforms.ToTensor()
    timg = tt(img)
    timg.unsqueeze(0)
    # writer.add_image('raw', timg, epoch)


def visualize_cos(feat, labels, epoch):
    colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
              '#ff00ff', '#990000', '#999900', '#009900', '#009999']

    fig = Figure(figsize=(6, 6), dpi=100)
    fig.clf()
    canvas = FigureCanvas(fig)
    ax = fig.gca()


    feat = feat / feat.norm(2, 1).unsqueeze(1).repeat(1, 2)
    feat = feat.data.cpu().numpy()
    labels = labels.data.cpu().numpy()

    weight = head.state_dict()['weight'].t()
    weight = weight / weight.norm(2, 1).unsqueeze(1).repeat(1, 2)
    weight = weight.data.cpu().numpy()

    for i in range(10):
        ax.scatter(feat[labels == i, 0], feat[labels == i, 1], c=colors[i], s=1)
        ax.text(feat[labels == i, 0].mean(), feat[labels == i, 1].mean(), str(i), color='black', fontsize=12)
        ax.plot([0,weight[i][0]],[0,weight[i][1]],linewidth=2,color=colors[i])
    ax.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    ax.text(0, 0, "epoch=%d" % epoch)
    canvas.draw()
    fig.savefig(imgDir + '/cos_epoch=%d.jpg' % epoch)
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    tt = transforms.ToTensor()
    timg = tt(img)
    timg.unsqueeze(0)
    # writer.add_image('raw_cos', timg, epoch)


def train_mnist(epoch, optimizer_nn):
    feature_list = []
    label_list = []
    train_loss_nn = 0
    train_acc = 0
    # EbeddingNet.train()  # 切换到训练模式
    for images, label in train_loader:
        print(type(images))
        print(type(label))
        # images = images.cuda()
        # label = label.cuda()
        images = images.to(device)
        label = label.to(device)
        # 前向传播
        feature= EbeddingNet(images, epoch)
        out = head(feature)

        feature_list.append(feature)
        label_list.append(label)
        loss = myCrossEntropyLoss(out, label)
        # 反向传播
        optimizer_nn.zero_grad()
        loss.backward()
        optimizer_nn.step()
        # 记录误差
        train_loss_nn += loss.item()  #############data[0]->item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = float(num_correct) / images.shape[0]
        train_acc += acc

    feats = torch.cat(feature_list, 0)
    labels = torch.cat(label_list, 0)

    visualize(feats, labels, epoch)
    visualize_cos(feats, labels, epoch)
    return train_loss_nn, train_acc


def tes_mnist(epoch):
    # 在测试集上检验效果
    with torch.no_grad():
        eval_loss = 0
        eval_acc = 0
        EbeddingNet.eval()  # 将模型改为预测模式
        with torch.no_grad():
            for images, label in test_loader:
                # images = images.cuda()
                # label = label.cuda()
                images = images.to(device)
                label = label.to(device)
                feature = EbeddingNet(images, epoch)
                out = head(feature)
                loss = myCrossEntropyLoss(out, label)
                eval_loss += loss.item()
                _, pred = out.max(1)
                num_correct = (pred == label).sum().item()
                acc = float(num_correct) / images.shape[0]
                eval_acc += acc
    return eval_acc, eval_loss

def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 5

def main():
    # 开始训练
    torch.manual_seed(1)
    optimizer_nn = optim.Adam([
        {'params': EbeddingNet.parameters(), 'lr': 1e-3},
        {'params': head.parameters(), 'lr': 1e-3},
    ])
    print(optimizer_nn.param_groups)
    milestone = [20, 40, 60]
    for epoch in tqdm(range(80)):
        if epoch in milestone:
            schedule_lr(optimizer_nn)
        train_loss_nn, train_acc = train_mnist(epoch, optimizer_nn)
        eval_acc, eval_loss = tes_mnist(epoch)

        # writer.add_scalar('Train Loss_nn', train_loss_nn / len(train_loader), epoch)
        # writer.add_scalar('Train Acc', train_acc / len(train_loader), epoch)
        # writer.add_scalar('eval Loss', eval_loss / len(test_loader), epoch)
        # writer.add_scalar('eval acc', eval_acc / len(test_loader), epoch)
        print(
            'epoch: {}, Train Loss_nn: {:.6f},  Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f},'
                .format(epoch, train_loss_nn / len(train_loader),
                        train_acc / len(train_loader),
                        eval_loss / len(test_loader), eval_acc / len(test_loader)))
    if not os.path.exists('./save_raw'):
        os.mkdir('./save_raw')
    torch.save(EbeddingNet, './save_raw/Raw.pkl')


if __name__ == '__main__':
    main()
