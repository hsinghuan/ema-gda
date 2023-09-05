import os
import torch
from torch.utils.data import Dataset, DataLoader

src_num = 50000
int_num = 400000
tgt_num = 50000
covertype_total_train_num = src_num + int_num + tgt_num
covertype_class_num = 7
tgt_test_num = 10000
interval = 25000
covertype_domains = list(range(1 + int_num // interval + 1)) # source, intermediate domains, target

class CovertypeDataset(Dataset):
    def __init__(self, data_dir, indexed=False):
        self.indexed = indexed
        self.data, self.target = [], []
        with open(os.path.join(data_dir, "covtype.data"), "r") as f:
            for line in f:
                line = line.strip().split(",")
                self.data.append([float(x) for x in line[:len(line)-1]])
                self.target.append(int(line[len(line)-1]) - 1)

        self.data = torch.tensor(self.data)
        self.target = torch.tensor(self.target)

        # Sort data by distance to water body
        dist_to_water = torch.linalg.vector_norm(self.data[:,3:5], dim=1) # 3 and 4 are distance to water
        sort_indices = torch.argsort(dist_to_water)
        self.data = self.data[sort_indices]
        self.target = self.target[sort_indices]
        # Standardize the first 10 dimensions according to train dataset
        mean_0_10 = torch.mean(self.data[:covertype_total_train_num,:10], dim=0)
        std_0_10 = torch.std(self.data[:covertype_total_train_num,:10], dim=0)
        self.data[:,:10] = (self.data[:,:10] - mean_0_10) / std_0_10


    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        if self.indexed:
            return idx, self.data[idx], self.target[idx]
        else:
            return self.data[idx], self.target[idx]

def get_covertype(data_dir: str, domain_idx: int, batch_size: int, target_test: bool = False, val: bool = True, indexed: bool = False):
    dataset = CovertypeDataset(data_dir=data_dir, indexed=indexed)

    if target_test:
        assert domain_idx == len(covertype_domains) - 1
        start_idx = src_num + int_num + tgt_num
        end_idx = start_idx + tgt_test_num
        print("start idx:", start_idx, "end idx:", end_idx)
        dataset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return test_loader

    if domain_idx == 0:
        start_idx = 0
        end_idx = src_num
    elif domain_idx == len(covertype_domains) - 1:
        start_idx = src_num + int_num
        end_idx = start_idx + tgt_num
    else:
        start_idx = src_num + (domain_idx - 1) * interval
        end_idx = src_num + domain_idx * interval
    print("start idx:", start_idx, "end idx:", end_idx)
    dataset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))
    if val:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9),
                                                                             len(dataset) - int(len(dataset) * 0.9)])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    else:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return train_loader

# print(covertype_domains)
# train_loader = get_covertype("/home/hhchung/data/covertype/", domain_idx=9, batch_size = 4, target_test = False, val = False, indexed = True)
# label_cnt = torch.zeros(7)
# for idx, x, y in train_loader:
#     label_cnt[y] += 1.
# print(label_cnt)