import os
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
import matplotlib.pyplot as plt



src_num = 2000
int_num = 12000
tgt_num = 2000
portraits_total_train_num = src_num + int_num + tgt_num
portraits_class_num = 2
tgt_test_num = 1000
interval = 2000
portraits_domains = list(range(1 + int_num // interval + 1)) # source, intermediate domains, target

class PortraitsDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None, indexed=False):
        self.transform = transform
        self.target_transform = target_transform
        self.indexed = indexed

        F_filenames = os.listdir(os.path.join(img_dir, "F"))
        M_filenames = os.listdir(os.path.join(img_dir, "M"))
        filenames = F_filenames + M_filenames
        img_labels = [0] * len(F_filenames) + [1] * len(M_filenames)
        filenames, img_labels = (list(t) for t in zip(*sorted(zip(filenames, img_labels))))
        img_paths = []
        for i in range(len(filenames)):
            gender_str = "F" if img_labels[i] == 0 else "M"
            img_paths.append(os.path.join(img_dir, gender_str, filenames[i]))

        self.img_paths = img_paths
        self.img_labels = img_labels

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
        # TODO: Indexing the training dataset for temporal ensembling
        if self.indexed:
            return idx, image, label
        else:
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
    return mean, std


def get_portraits(data_dir: str, domain_idx: int, batch_size: int, target_test: bool = False, val: bool = True, indexed: bool = False):
    transform = transforms.Compose([transforms.Normalize((128.8960,), (62.1401,)), transforms.Resize((128,128))])
    dataset = PortraitsDataset(data_dir, transform=transform, indexed=indexed)
    if target_test:
        assert domain_idx == len(portraits_domains) - 1
        start_idx = src_num + int_num + tgt_num
        end_idx = start_idx + tgt_test_num
        print("start idx:", start_idx, "end idx:", end_idx)
        dataset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return test_loader

    if domain_idx == 0:
        start_idx = 0
        end_idx = src_num
    elif domain_idx == len(portraits_domains) - 1:
        start_idx = src_num + int_num
        end_idx = start_idx + tgt_num
    else:
        start_idx = src_num + (domain_idx - 1) * interval
        end_idx = src_num + domain_idx * interval
    print("start idx:", start_idx, "end idx:", end_idx)
    dataset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))
    if val:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    else:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return train_loader




# dataset = PortraitsDataset("/home/hhchung/data/faces_aligned_small_mirrored_co_aligned_cropped_cleaned", portraits_domains[0], indexed=True)
# print(len(dataset))
# print(dataset[3], dataset[3][0].shape)
# plt.imshow(dataset[3][0].squeeze(0), cmap='gray')
# plt.show()

# mean, std = compute_portraits_stats("/home/hhchung/data/faces_aligned_small_mirrored_co_aligned_cropped_cleaned")
# print(mean, std)

# train_loader, val_loader = get_portraits("/home/hhchung/data/faces_aligned_small_mirrored_co_aligned_cropped_cleaned", 0, 256)
# print(len(train_loader))
# print(len(val_loader))
# for data, y in train_loader:
#     print(data.shape)
#     print(y.shape)
#     break

# test_loader = get_portraits("/home/hhchung/data/faces_aligned_small_mirrored_co_aligned_cropped_cleaned", portraits_domains[-1], 128, val=False, target_test=True)
# for data, y in test_loader:
#     print(data.shape)
#     print(y.shape)

# train_loader, val_loader = get_portraits("/home/hhchung/data/faces_aligned_small_mirrored_co_aligned_cropped_cleaned", 0, 256, val=True, indexed=True)
# print(len(train_loader))
# print(len(val_loader))
# for idx, data, y in val_loader:
#     print(idx)
#     break