import os
import sys
import copy
import logging
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.FedAvg import FedAvg
from dataset.get_dataset import get_datasets
from utils.weight_perturbation import WPOptim
from val import compute_bacc, compute_loss_of_classes
from networks.networks import efficientb0
from utils.local_training import LocalUpdate
from utils.utils import set_seed, TensorDataset, classify_label
from utils.sample_dirichlet import clients_indices
from tqdm import tqdm


def args_parser():  # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='isic2019', help='dataset name')  # 数据集
    parser.add_argument('--exp', type=str,
                        default='FedIIC', help='experiment name')  # 实验名称
    parser.add_argument('--batch_size', type=int,
                        default=2, help='batch_size per gpu')  # 每个gpu分配的batch size（batch * batch_size = 样本数量）
    parser.add_argument('--base_lr', type=float,
                        default=3e-4, help='base learning rate')  # 初始学习率
    parser.add_argument('--alpha', type=float,
                        default=1.0, help='parameter for non-iid')  # 对非独立同分布的数据的参数
    parser.add_argument('--k1', type=float,
                        default=2.0, help='weight for Intra-client contrastive learning')  # 客户内部的对比学习权重k1
    parser.add_argument('--k2', type=float,
                        default=2.0, help='weight for Inter-client contrastive learning')  # 客户内部的对比学习权重k2
    parser.add_argument('--d', type=float,
                        default=0.25, help='difficulty')  # 数据集的困难程度
    parser.add_argument('--deterministic', type=int,
                        default=1, help='whether use deterministic training')  # 确定学习（同输入同输出）
    parser.add_argument('--seed', type=int,
                        default=0, help='random seed')  # 随机种子
    parser.add_argument('--gpu', type=str,
                        default='0', help='GPU to use')  # gpu序号
    parser.add_argument('--local_ep', type=int,
                        default=1, help='local epoch')  # 本地模型完整遍历数据集的周期数
    parser.add_argument('--rounds', type=int, default=200, help='rounds')


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # ------------------------------ deterministic or not ------------------------------
    # 是否确保在相同的初始条件下，每次训练都会得到相同的结果，旨在消除随机性，使模型的训练过程是可重复和可预测
    if args.deterministic:
        cudnn.benchmark = False  # 指示CuDNN是否应根据硬件条件选择最快的算法
                                 # 如果设置为True，CuDNN会在每次调用时根据当前硬件条件选择最优的卷积算法
                                 # 这会带来一定的随机性
        cudnn.deterministic = True  # 指示CuDNN是否应该以确定性模式运行
                                    # 如果设置为True，CuDNN会使用确定性算法执行卷积操作
                                    # 以确保在相同的输入数据和权重情况下得到相同的输出
                                    # 这种设置会牺牲一定的性能，但能够确保结果的一致性
        set_seed(args)  # 使用超参数中的seed设置各模块的seed参数，使得随机性是可重复的

    # ------------------------------ output files ------------------------------
    outputs_dir = 'outputs2'
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)
    exp_dir = os.path.join(outputs_dir,
                           args.exp + '_' + '_' + str(args.local_ep) + '_' + str(args.k1) + '_' + str(args.k2))
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    models_dir = os.path.join(exp_dir, 'models')  # 保存的模型目录
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    logs_dir = os.path.join(exp_dir, 'logs')  # 保存的训练输出日志目录
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    tensorboard_dir = os.path.join(exp_dir, 'tensorboard')  # 保存的可用于可视化的训练记录目录
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)

    logging.basicConfig(filename=logs_dir + '/logs.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    # 添加了一个处理器（handler），将日志输出到标准输出流（stdout），即控制台
    # 这样可以在运行代码时，将日志信息同时输出到 控制台 和 日志文件
    # 如果不指定日志输出目录，则默认为控制台
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # 将参数args的信息记录到日志中，方便后续回溯和调试
    logging.info(str(args))

    # 创建一个SummaryWriter对象，用于将训练过程中的信息写入TensorBoard日志文件，以便进行可视化分析和监控训练过程
    # TensorBoard是TensorFlow提供的一个可视化工具，可以用于查看模型的训练曲线、模型结构、数据分布等信息
    writer = SummaryWriter(tensorboard_dir)

    # ------------------------------ dataset and dataloader ------------------------------
    train_dataset, val_dataset, test_dataset = get_datasets(args)  # 通过get_datasets函数传递预处理后的数据集
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=4)
    # val_loader负责批量加载验证数据集，并提供一些额外的功能，比如数据的随机化、分批处理等
    # 它是将验证数据集有效地传送给模型进行验证的重要工具，简化了数据加载的过程，并提供了一些额外的功能来提高训练效率

    if args.dataset == "isic2019":
        args.n_clients = 10
    elif args.datasets == "ich":
        args.n_clients = 20
    else:
        raise

    # ------------------------------ global and local settings ------------------------------
    # 获取训练数据集中的类别数量，通常用于初始化模型的输出层，以确保输出层的维度与数据集的类别数量相匹配
    n_classes = train_dataset.n_classes
    # 创建一个全局模型 net_glob，使用名为 efficientb0 的模型（可能是一个基于EfficientNet的模型），并将其移动到GPU进行训练
    # n_classes 参数指定了输出层的类别数量，args 参数可能包含了其他模型初始化所需的参数
    net_glob = efficientb0(n_classes=n_classes, args=args).cuda()
    # 将全局模型设置为训练模式，这意味着在模型的前向传播过程中会启用一些特定于训练的操作，比如 dropout
    net_glob.train()
    # 获取了全局模型的参数（权重和偏置），并将它们存储在 w_glob 中
    w_glob = net_glob.state_dict()
    # 初始化存储本地模型参数、本地训练器（optimizer）和本地模型的列表，这些列表将用于存储每个客户端的模型和相关的训练器
    w_locals = []
    trainer_locals = []
    net_locals = []
    # 创建了一个列表，其中包含了所有客户端的 ID
    # args.n_clients 可能是作为参数传递给代码的一个值，用于指定客户端的数量
    user_id = list(range(args.n_clients))

    # Here, we follow CreFF (https://arxiv.org/abs/2204.13399).
    # （缓解长尾数据FL）
    # 调用了一个函数 classify_label
    # 该函数将训练数据集中的样本按其类别进行分类，并返回一个字典，其中键是类别标签，值是属于该类别的样本的索引列表
    list_label2indices = classify_label(train_dataset.targets, n_classes)
    # 调用了一个函数 clients_indices，该函数使用类别标签和其他参数（如客户端数量、alpha值和随机种子）来创建一个字典
    # 该字典表示了每个客户端用户应该拥有的样本索引
    # 该函数可能根据参数对样本进行了分配，以确保每个客户端的样本数量大致相等
    # 并且每个客户端的样本中包含各个类别的样本，以保持数据的多样性
    dict_users = clients_indices(list_label2indices, n_classes, args.n_clients, args.alpha, args.seed)
    # 创建了一个列表 dict_len，其中包含了每个客户端用户的样本数量
    # 通过对每个客户端的样本数量进行统计，可以了解每个客户端的数据负载情况，以便在联邦学习中进行调整和优化
    dict_len = [len(dict_users[id]) for id in user_id]

    # 可能是为了模拟每个客户端在本地训练时使用的全局模型的副本，以便在联合学习中进行分布式训练时使用
    for id in user_id:
        trainer_locals.append(LocalUpdate(args, id, copy.deepcopy(train_dataset), dict_users[id]))
        w_locals.append(copy.deepcopy(w_glob))
        net_locals.append(copy.deepcopy(net_glob).cuda())

    images_all = {}
    labels_all = {}

    # for id in user_id:  # to compute loss quickly
    #     local_set = copy.deepcopy(trainer_locals[id].local_dataset)
    #     images_all[id] = torch.cat([torch.unsqueeze(local_set[i][0][0], dim=0)
    #                                 for i in range(len(local_set))])
    #     labels_all[id] = torch.tensor([int(local_set[i][1])
    #                                    for i in range(len(local_set))]).long()
    #     print(id, ':', len(images_all[id]), labels_all[id])
    # torch.set_printoptions(threshold=np.inf)  # 打印张量不省略，即使很大也全部显示

    # for id in user_id:  # to compute loss quickly
    #     local_set = copy.deepcopy(trainer_locals[id].local_dataset)
    #     images_all[id] = []
    #     labels_all[id] = []
    #
    #     # 使用tqdm包装内层循环，显示动态进度条
    #     for i in tqdm(range(len(local_set)), desc=f'Processing user {id}'):
    #         images_all[id].append(torch.unsqueeze(local_set[i][0][0], dim=0))
    #         labels_all[id].append(int(local_set[i][1]))
    #
    #     images_all[id] = torch.cat(images_all[id])
    #     labels_all[id] = torch.tensor(labels_all[id]).long()
    #     print(id, ':', len(images_all[id]), labels_all[id])

    # ------------------------------ begin training ------------------------------
    set_seed(args)
    best_performance = 0.  # 最优性能
    lr = args.base_lr      # 学习率
    acc = []               # 预测准确率（对于test_dataset）
    # 测试集和验证集在机器学习和深度学习中有不同的作用，因此两者都是必要的。它们的主要区别在于它们在训练过程中的使用方式和目的。

    # 验证集：
    # 作用：验证集用于在训练过程中评估模型在未见过的数据上的性能。它主要用于模型选择、超参数调优和防止过拟合。
    # 使用方式：在每次迭代或周期结束后，模型会在验证集上进行评估，以确定当前模型的性能。
    # 数据量：通常情况下，验证集的大小介于训练集和测试集之间，通常比测试集稍小。

    # 测试集：
    # 作用：测试集用于最终评估模型的性能，并提供模型在真实应用场景下的预测效果。
    # 使用方式：在模型训练完成后，使用测试集对模型进行最终评估，得出模型的准确率、精确度、召回率等指标。
    # 数据量：测试集的数据量通常是验证集的大小或更大，以更全面地评估模型在真实场景中的表现。

    # 总的来说，验证集和测试集在训练过程中的作用和使用方式不同，
    # 验证集用于训练过程中的模型选择和调优，而测试集则用于最终评估模型的性能。

    for com_round in range(args.rounds):
        logging.info(f'\n======================> round: {com_round} <======================')  # 写日志文件
        loss_locals = []
        writer.add_scalar('train/lr', lr, com_round)  # 将学习率记录到TensorBoard训练日志中，跟踪学习率的变化

        with torch.no_grad():
            # 从全局模型的参数字典中提取模型的最后一层（全连接层）的权重参数，并将其移动到GPU上
            # detach()方法创建了一个新的张量，且不保留梯度信息，clone()方法复制了这个张量的值，避免对原始参数的修改
            class_embedding = w_glob["model._fc.weight"].detach().clone().cuda()

            # 将全局模型的最后一层权重参数输入到net_glob模型中的projector方法中，得到一个特征均值向量，并将其移动到GPU上
            # 同样地，也使用了detach()和clone()方法。
            feature_avg = net_glob.projector(class_embedding).detach().clone()

        # 通过计算特征向量的余弦相似度来评估模型的特征均匀性。
        print("similarity before")
        print(torch.matmul(F.normalize(feature_avg, dim=1),
                           F.normalize(feature_avg, dim=1).T))
        logging.info("similarity before")
        logging.info(torch.matmul(F.normalize(feature_avg, dim=1),
                                  F.normalize(feature_avg, dim=1).T))

        # 特征均值向量
        feature_avg.requires_grad = True
        # 定义优化器调整特征均值向量，以改进相似度
        # optimizer_f = torch.optim.SGD([feature_avg], lr=0.1)  # ===
        optimizer_f = torch.optim.SGD([feature_avg], lr=0.1)  # ===

        mask = torch.ones((n_classes, n_classes)) - torch.eye((n_classes))
        mask = mask.cuda()
        for i in range(1000):
            feature_avg_n = F.normalize(feature_avg, dim=1)
            cos_sim = torch.matmul(feature_avg_n, feature_avg_n.T)
            cos_sim = ((cos_sim * mask).max(1)[0]).sum()
            optimizer_f.zero_grad()
            cos_sim.backward()
            optimizer_f.step()
        print("similarity after")
        print(torch.matmul(F.normalize(feature_avg, dim=1),
                           F.normalize(feature_avg, dim=1).T))
        logging.info("similarity after")
        logging.info(torch.matmul(F.normalize(feature_avg, dim=1),
                                  F.normalize(feature_avg, dim=1).T))

        # 初始化 loss、类别数、全局模型移动
        loss_matrix = torch.zeros(args.n_clients, n_classes)  # 每个客户端的损失
        class_num = torch.zeros(args.n_clients, n_classes)  # 类别数量
        net_glob = net_glob.cuda()  # 将全局模型移动到GPU

        # 客户端 本地模型 的 loss 计算
        # for id in tqdm(user_id, desc=f'Processing round {com_round}'):  # 使用tqdm包装用户ID循环，显示动态进度条
        #     class_num[id] = torch.tensor(trainer_locals[id].local_dataset.get_num_class_list())
        #     dataset_client = TensorDataset(images_all[id], labels_all[id])
        #     dataLoader_client = DataLoader(dataset_client, batch_size=32, shuffle=False)
        #     loss_matrix[id] = compute_loss_of_classes(net_glob, dataLoader_client, n_classes)
        # num = torch.sum(class_num, dim=0, keepdim=True)
        # logging.info("class-num of this round")
        # logging.info(num)
        # loss_matrix = loss_matrix / (1e-5 + num)
        loss_class = torch.sum(loss_matrix, dim=0)
        # logging.info("loss of this round")
        # logging.info(loss_class)

        # 定义: 加入了 权重扰动 的本地训练优化器（将 SGD 改为了 Adam）
        # optimizers = [WPOptim(params=net_locals[idx].parameters(), base_optimizer=torch.optim.Adam, lr=1e-3,
        #                       alpha=0.05, weight_decay=1e-4) for idx in range(args.n_clients)]
        # 改为 5e-4 ？？？

        # local training 本地训练
        for id in user_id:
            # 对每个客户端进行循环，设置学习率、损失类别
            trainer_locals[id].lr = lr
            local = trainer_locals[id]
            local.loss_class = loss_class
            #local.loss_class.requires_grad = True#
            net_local = net_locals[id]
            #net_local.requires_grad = True
            # optimizer = optimizers[id]  # 加入 权重扰动 优化器

            # 进行本地模型的 FedIIC 训练，保存每个客户端的权重和损失，并使用加入了 权重扰动 的训练优化器
            w, loss = local.train_FedIIC(copy.deepcopy(net_local), copy.deepcopy(feature_avg), writer)
            w_locals[id] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))

        # upload and download 全局模型聚合
        with torch.no_grad():
            w_glob = FedAvg(w_locals, dict_len)  # 使用 FedAvg 算法进行全局模型的权重聚合
        net_glob.load_state_dict(w_glob)
        for id in user_id:
            net_locals[id].load_state_dict(w_glob)  # 更新全局模型和每个客户端的模型权重

        # global validation 全局验证
        net_glob = net_glob.cuda()
        ## 使用全局模型在验证集上进行验证，计算并记录验证的准确度和混淆矩阵
        bacc_g, conf_matrix = compute_bacc(net_glob, val_loader, get_confusion_matrix=True, args=args)
        writer.add_scalar( f'glob/bacc_val', bacc_g, com_round)
        logging.info('global conf_matrix')
        logging.info(conf_matrix)

        # save model
        if bacc_g > best_performance:  # 如果当前模型的验证准确度超过历史最佳准确度，则保存该模型权重
            best_performance = bacc_g
            torch.save(net_glob.state_dict(), models_dir +
                       f'/best_model_{com_round}_{best_performance}.pth')
            torch.save(net_glob.state_dict(), models_dir + '/best_model.pth')
        logging.info(f'best bacc: {best_performance}, now bacc: {bacc_g}')
        acc.append(bacc_g)

    writer.close()
    logging.info(acc)
