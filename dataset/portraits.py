import os
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
import matplotlib.pyplot as plt

portraits_domains = [[1905,1940],[1940,1950],[1950,1960],[1960,1970],[1970,1980],[1980,1990],[1990,2000],[2000,2014]]

class PortraitsDataset(Dataset):
    def __init__(self, img_dir, domain, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        F_paths = [os.path.join(img_dir, "F", filename) for filename in os.listdir(os.path.join(img_dir, "F")) if int(filename[:4]) >= domain[0] and int(filename[:4]) < domain[1]]
        M_paths = [os.path.join(img_dir, "M", filename) for filename in os.listdir(os.path.join(img_dir, "M")) if int(filename[:4]) >= domain[0] and int(filename[:4]) < domain[1]]
        self.img_paths = F_paths + M_paths
        self.img_labels = [0] * len(F_paths) + [1] * len(M_paths)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path, ImageReadMode.GRAY).float()
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def compute_portraits_stats(img_dir: str):
    full_domain = [1905, 2014]
    dataset = PortraitsDataset(img_dir, full_domain)
    loader = DataLoader(dataset, batch_size=256)
    mean, std, nb_samples = 0., 0., 0
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.float().mean(2).sum(0)
        std += data.float().std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    print(nb_samples)
    return mean, std


def get_portraits(data_dir: str, domain: List, batch_size: int):
    transform = transforms.Compose([transforms.Normalize((128.8960,), (62.1401,)), transforms.Resize((32,32))])
    dataset = PortraitsDataset(data_dir, domain, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# dataset = PortraitsDataset("/home/hhchung/data/faces_aligned_small_mirrored_co_aligned_cropped_cleaned", portraits_domains[0])
# print(dataset[3], dataset[3][0].shape)
# plt.imshow(dataset[3][0].squeeze(0), cmap='gray')
# plt.show()

# mean, std = compute_portraits_stats("/home/hhchung/data/faces_aligned_small_mirrored_co_aligned_cropped_cleaned")
# print(mean, std)

# train_loader, val_loader = get_portraits("/home/hhchung/data/faces_aligned_small_mirrored_co_aligned_cropped_cleaned", portraits_domains[0], 256)
# print(len(train_loader))
# print(len(val_loader))
# for data, y in train_loader:
#     print(data.shape)
#     print(y.shape)
#     break