import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


def compute_bacc(model, dataloader, get_confusion_matrix, args):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for (x, label) in dataloader:
            x = x.cuda()
            _, logits = model(x)
            pred = torch.argmax(logits, dim=1)

            all_preds.append(pred.cpu())
            all_labels.append(label)

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    acc = balanced_accuracy_score(all_labels, all_preds)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(all_labels, all_preds)

    if get_confusion_matrix:
        return acc, conf_matrix
    else:
        return acc



def compute_loss(model, dataloader):
    # 计算所有样本的总损失

    criterion = nn.CrossEntropyLoss()  # 对所有的样本计算平均的交叉熵损失
    model.eval()
    loss = 0.
    with torch.no_grad():
        for (x, label) in dataloader:
            if isinstance(x, list):
                x = x[0]
            x, label = x.cuda(), label.cuda()
            _, logits = model(x)
            loss += criterion(logits, label)
    return loss


def compute_loss_of_classes(model, dataloader, n_classes):
    # 计算每个类别的总损失

    criterion = nn.CrossEntropyLoss(reduction="none")  # 初始化损失函数
    # 为每个样本计算一个独立的交叉熵损失，并未对这些损失进行平均
    model.eval()  # 将模型设置为评估模式，模型在这个过程中不会更新梯度，也不会进行Dropout或Batch Normalization的操作

    # 初始化损失相关变量
    loss_class = torch.zeros(n_classes).float()  # 每个类别的总损失
    loss_list = []  # 每个样本的损失
    label_list = []  # 每个样本的标签

    with torch.no_grad():
        for (x, label) in dataloader:
            # 处理输入数据
            if isinstance(x, list):
                x = x[0]
            x, label = x.cuda(), label.cuda()  # 如果输入数据x是列表类型，取第一个元素，然后将x和label移到GPU上
            _, logits = model(x)  # 模型前向传播
            loss = criterion(logits, label)  # 交叉熵损失函数计算预测的logits和真实标签之间的损失
            loss_list.append(loss)  # 存储损失和标签
            label_list.append(label)

    # 合并损失和标签列表
    loss_list = torch.cat(loss_list).cpu()
    label_list = torch.cat(label_list).cpu()

    # 计算每个类别的总损失
    for i in range(n_classes):
        idx = torch.where(label_list == i)[0]
        loss_class[i] = loss_list[idx].sum()

    return loss_class
