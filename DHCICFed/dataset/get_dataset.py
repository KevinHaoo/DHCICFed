import logging
from collections import Counter
from torchvision.transforms import transforms

from dataset.dataset import SkinDataset, ichDataset
from dataset.randaugment import rand_augment_transform


def get_datasets(args):
    # 在函数内部，首先定义了数据预处理的一系列操作
    # 包括了训练时的数据增强操作（augmentation）和验证、测试时的标准化操作（normalization）
    # 这些操作包括了随机裁剪、水平翻转、随机色彩变换、随机灰度变换等，目的是增加训练数据的多样性，提高模型的泛化能力
    #
    # 然后，根据定义的数据预处理操作，分别创建了训练、验证和测试数据集
    # 其中，训练数据集使用了两种不同的数据预处理操作，分别是trans和augmentation_sim
    # 验证和测试数据集使用了相同的标准化操作val_transform。
    #
    # 最后，通过创建DataLoader对象，将创建好的数据集传入
    # 设置了批量大小（batch size）、是否打乱数据（shuffle）、并行加载数据的线程数（num_workers）等参数
    # 从而得到了训练、验证和测试的数据加载器
    # 这些数据加载器可以在模型训练、验证和测试时被调用，用于批量加载数据，进而进行模型的训练和评估
    if args.dataset == "isic2019":
        root = "/root/autodl-tmp/data/isic2019classification"
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(
            224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
        trans = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform(
                'rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
            transforms.ToTensor(),
            normalize,
        ])
        augmentation_sim = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        train_dataset = SkinDataset(root=root, mode="train",
                                    transform=[trans, augmentation_sim])
        val_dataset = SkinDataset(root=root, mode="valid",
                                  transform=val_transform)
        test_dataset = SkinDataset(root=root, mode="test",
                                   transform=val_transform)

    elif args.dataset == "ich":
        root = "/root/autodl-tmp/data/ICH"
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])

        trans1 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        trans2 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

        train_dataset = ichDataset(
            root=root, mode="train", transform=[trans1, trans2])
        val_dataset = ichDataset(
            root=root, mode="valid", transform=val_transform)
        test_dataset = ichDataset(
            root=root, mode="test", transform=val_transform)

    else:
        raise

    logging.info(Counter(train_dataset.targets))
    logging.info(Counter(val_dataset.targets))
    logging.info(Counter(test_dataset.targets))

    return train_dataset, val_dataset, test_dataset

