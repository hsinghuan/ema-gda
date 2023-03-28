from typing import List
import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF


class RandomRotation:
    def __init__(self, range:List):
        self.range = range

    def __call__(self, x):
        angle = np.random.uniform(self.range[0], self.range[1])
        return TF.rotate(x, angle)


rotate_mnist_domains = [[0,5],[5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45],[45,50],[50,55],[55,60]]
degree_sum = 60

def get_rotate_mnist(data_dir: str, domain: List, batch_size: int, target_test: bool = False, val: bool = True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), RandomRotation(domain)])
    if target_test:
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    train_dataset = torch.utils.data.Subset(train_dataset, range(len(train_dataset) * domain[0] // degree_sum, len(train_dataset) * domain[1] // degree_sum)) # each image will be seen only once
    if val:
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset) * 0.9), len(train_dataset) - int(len(train_dataset) * 0.9)])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader