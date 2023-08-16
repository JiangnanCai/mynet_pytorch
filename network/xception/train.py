
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms, datasets
import os
import torch.optim as optim
from backbone.xception import Xception

# gpu设备 ---------------------------------------------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据处理 ---------------------------------------------------------------------------------------------------------------
# 在训练过程中的数据增强
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])d
}

# 指定数据集路径
data_root = r'/home/cai/Documents/Inception/archive(1)'
image_path = os.path.join(data_root, 'dataset')

# 转成数据类，加载数据类
batch_size = 4

train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'), transform=data_transform['train'])
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'valid'), transform=data_transform['val'])
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0)

# 训练参数部分 -----------------------------------------------------------------------------------------------------------
net = Xception(num_classes=5)
net.to(device)

start_lr = 0.045


def adjust_lr(optimizer, epoch, start_lr):
    if epoch % 2 == 0:
        lr = start_lr * 0.96
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


loss_function = CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.0003)
optimizer = optim.SGD(net.parameters(), lr=start_lr)

best_acc = 0.0
save_path = 'xception.pth'
val_num = len(validate_dataset)
for epoch in range(100):
    net.train()
    adjust_lr(optimizer, epoch, start_lr)
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss_0 = loss_function(logits, labels.to(device))
        loss = loss_0
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:.3f} | {:^3.0f}%[{}->{}] ".format(loss, int(rate * 100), a, b), end="")
    net.eval()
    acc = 0.0
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('\n[epoch %d] train_loss: %.3f  test accuracy: %.3f' % (epoch + 1, running_loss / step, best_acc))
print("finish training")

