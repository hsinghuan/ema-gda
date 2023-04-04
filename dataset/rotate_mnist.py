from typing import List
import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


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
interval = 2000
interval_num = int_num // interval
separated_int_domain = [[int_domain[0] + i * (int_domain[1] - int_domain[0]) // interval_num,
                    int_domain[0] + (i + 1) * (int_domain[1] - int_domain[0]) // interval_num] for i in range(interval_num)]
rotate_mnist_domains = [src_domain] + separated_int_domain + [tgt_domain]

def rotate(dataset, domain, continual):
    rotated_data_list = []
    y_list = []
    data_num = len(dataset)
    for i, (data, y) in enumerate(dataset):
        if continual:
            angle = i * (domain[1] - domain[0]) / data_num + domain[0]
        else:
            angle = np.random.uniform(domain[0], domain[1])
        rotated_data_list.append(TF.rotate(data, angle))
        y_list.append(y)
    rotated_data_tensor = torch.cat(rotated_data_list).unsqueeze(1)
    rotated_y_tensor = torch.tensor(y_list)
    dataset = torch.utils.data.TensorDataset(rotated_data_tensor, rotated_y_tensor)
    return dataset

def get_rotate_mnist(data_dir: str, domain_idx: int, batch_size: int, target_test: bool = False, val: bool = True, inter_idx: int = None):
    domain = rotate_mnist_domains[domain_idx]
    print("Domain:", domain)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) #, RandomRotation(domain)])
    if target_test:
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        test_dataset = rotate(test_dataset, domain, continual=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader

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

    train_dataset = rotate(train_dataset, domain, continual=domain_idx!=0)
    # print(f"Number of samples: {len(train_dataset)}")
    if val:
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset) * 0.9), len(train_dataset) - int(len(train_dataset) * 0.9)])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader


# train_loader = get_rotate_mnist("/home/hhchung/data/", 0, 256, target_test=False, val=False)
# train_loader = get_rotate_mnist("/home/hhchung/data/", 1, 256, target_test=False, val=False)
# train_loader = get_rotate_mnist("/home/hhchung/data/", len(rotate_mnist_domains) - 1, 256, target_test=False, val=False)
# for data, _ in train_loader:
#     img = data[30]
#     print(img.shape)
#     plt.imshow(img.squeeze(0), cmap='gray')
#     plt.show()
#     break