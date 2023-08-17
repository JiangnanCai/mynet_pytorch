import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset import Voc2012
from backbone.yolov1_backbone import YoloV1Backbone
from net import YoloV1
from loss import Yolov1Loss

from torch.utils.tensorboard import SummaryWriter


import os
import numpy as np
import math

# 检查gpu
use_gpu = torch.cuda.is_available()
assert use_gpu, 'no gpu'

print_freq = 5
tb_log_freq = 5

init_lr = 0.001
base_lr = 0.01
momentum = 0.9
weight_decay = 5.0e-4
num_epoch = 200
batch_size = 16
num_workers = 4


def update_lr(optimizer, epoch, burning_base, burning_exp=4.0):
    if epoch == 0:
        lr = init_lr + (base_lr - init_lr) * math.pow(burning_base, burning_exp)
    elif epoch == 1:
        lr = base_lr
    elif epoch == 75:
        lr = 0.001
    elif epoch == 105:
        lr = 0.0001
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / 10


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


backbone = YoloV1Backbone(conv_only=True, bn=True, init_weight=True)
backbone.features = torch.nn.DataParallel(backbone.features)

yolov1 = YoloV1(backbone.features)
yolov1.conv_layers = torch.nn.DataParallel(yolov1.conv_layers)
yolov1.cuda()

criterion = Yolov1Loss(num_grids=yolov1.num_grids)
optimizer = torch.optim.SGD(yolov1.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)

train_dataset = Voc2012(is_train=True, keyword='train')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_dataset = Voc2012(is_train=True, keyword='val')
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

print("number of train images: ", len(train_dataset))

best_val_loss = np.inf

for epoch in range(num_epoch):
    print('\n')
    print(f"Starting epoch {epoch} / {num_epoch}")

    yolov1.train()
    total_loss = 0.0
    total_batch = 0

    for i, (imgs, inputs) in enumerate(train_dataloader):
        update_lr(optimizer, epoch, float(i) / float(len(train_dataloader) - 1))
        lr = get_lr(optimizer)

        batch_size_this_iter = imgs.size(0)
        imgs, inputs = Variable(imgs).cuda(), Variable(inputs).cuda()

        preds = yolov1(imgs)
        loss = criterion(preds, inputs)
        loss_this_iter = loss.item()
        total_loss += loss_this_iter * batch_size_this_iter
        total_batch += batch_size_this_iter

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % print_freq == 0:
            print(f"Epoch: [{epoch}/{num_epoch}], Iter: [{i}/{len(train_dataloader)}], LR: {lr}, Loss: {loss_this_iter},"
                  f" Average Loss: {total_loss/float(total_batch)}")

    yolov1.eval()
    val_loss = 0.0
    total_batch = 0

    for i, (imgs, inputs) in enumerate(val_dataloader):
        batch_size_this_iter = imgs.size(0)
        imgs, inputs = Variable(imgs).cuda(), Variable(inputs).cuda()

        with torch.no_grad():
            preds = yolov1(imgs)
        loss = criterion(preds, inputs)
        loss_this_iter = loss.item()
        val_loss += loss_this_iter * batch_size_this_iter
        total_batch += batch_size_this_iter
    val_loss /= float(total_batch)

    torch.save(yolov1.state_dict(), './model_latest.pth')
    if best_val_loss > val_loss:
        best_val_loss = val_loss
        torch.save(yolov1.state_dict(), './model_best.pth')

    print(f"Epoch [{epoch}/{num_epoch}], Val loss: {val_loss}, Best val loss: {best_val_loss}")




