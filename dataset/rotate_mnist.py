from typing import List
import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

class IndexedMNIST(datasets.MNIST): # Use this for temporal ensembling
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(IndexedMNIST, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, index):
        image, label = super(IndexedMNIST, self).__getitem__(index) # may be wrong?
        return index, image, label

    def __len__(self):
        return super(IndexedMNIST, self).__len__()


class RandomRotation:
    def __init__(self, range:List):
        self.range = range

    def __call__(self, x):
        angle = np.random.uniform(self.range[0], self.range[1])
        return TF.rotate(x, angle)


src_domain = [0,5]
int_domain = [5,55]
tgt_domain = [55,60]
# rotate_mnist_domains = [[0,5], [5,55], [55,60]]
degree_sum = 60
src_num = 6000
int_num = 52000
tgt_num = 2000
rotate_mnist_total_train_num = src_num + int_num + tgt_num
rotate_mnist_class_num = 10
interval = 2000
interval_num = int_num // interval
separated_int_domain = [[int_domain[0] + i * (int_domain[1] - int_domain[0]) // interval_num,
                    int_domain[0] + (i + 1) * (int_domain[1] - int_domain[0]) // interval_num] for i in range(interval_num)]
rotate_mnist_domains = [src_domain] + separated_int_domain + [tgt_domain]

def rotate(dataset, domain, continual, indexed=False):
    index_list = [] if indexed else None
    rotated_img_list = []
    y_list = []
    data_num = len(dataset)

    for i, data in enumerate(dataset):
        if indexed:
            index, img, y = data
        else:
            img, y = data

        if continual:
            angle = i * (domain[1] - domain[0]) / data_num + domain[0]
        else:
            angle = np.random.uniform(domain[0], domain[1])

        if indexed:
            index_list.append(index)
        rotated_img_list.append(TF.rotate(img, angle))
        y_list.append(y)

    rotated_index_tensor = torch.tensor(index_list) if indexed else None
    rotated_img_tensor = torch.cat(rotated_img_list).unsqueeze(1)
    rotated_y_tensor = torch.tensor(y_list)
    if indexed:
        dataset = torch.utils.data.TensorDataset(rotated_index_tensor, rotated_img_tensor, rotated_y_tensor)
    else:
        dataset = torch.utils.data.TensorDataset(rotated_img_tensor, rotated_y_tensor)
    return dataset

def get_rotate_mnist(data_dir: str, domain_idx: int, batch_size: int, target_test: bool = False, val: bool = True, indexed: bool = False, train_shuffle: bool = True):
    domain = rotate_mnist_domains[domain_idx]
    print("Domain:", domain)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) #, RandomRotation(domain)])
    if target_test:
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        test_dataset = rotate(test_dataset, domain, continual=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader

    if indexed:
        train_dataset = IndexedMNIST(data_dir, train=True, download=True, transform=transform)
    else:
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)

    # print("len train dataset:", len(train_dataset))
    if domain_idx == 0: # == rotate_mnist_domains[0]:
        start_idx = 0
        end_idx = src_num
    elif domain_idx == len(rotate_mnist_domains) - 1: # == rotate_mnist_domains[-1]:
        start_idx = src_num + int_num
        end_idx = 60000
    else:
        start_idx = src_num + (domain_idx - 1) * interval
        end_idx = src_num + domain_idx * interval

    # print(f"Start Idx: {start_idx} End Idx: {end_idx}")
    train_dataset = torch.utils.data.Subset(train_dataset, range(start_idx, end_idx)) # each image will be seen only once

    train_dataset = rotate(train_dataset, domain, continual=domain_idx!=0, indexed=indexed)
    # print(f"Number of samples: {len(train_dataset)}")
    if val:
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset) * 0.9), len(train_dataset) - int(len(train_dataset) * 0.9)])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
        return train_loader


