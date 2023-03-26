from typing import List
import os


def process(data_dir: str, target_domain: List):
    import random
    random.seed(42)
    F_paths = [os.path.join("F", filename) for filename in os.listdir(os.path.join(data_dir, "F")) if
               int(filename[:4]) >= target_domain[0] and int(filename[:4]) < target_domain[1]]
    M_paths = [os.path.join("M", filename) for filename in os.listdir(os.path.join(data_dir, "M")) if
               int(filename[:4]) >= target_domain[0] and int(filename[:4]) < target_domain[1]]
    F_num = len(F_paths)
    M_num = len(M_paths)
    F_idx = list(range(F_num))
    M_idx = list(range(M_num))
    random.shuffle(F_idx)
    random.shuffle(M_idx)

    test_ratio = 0.2
    F_test_num = int(F_num * test_ratio)
    M_test_num = int(M_num * test_ratio)
    F_test_paths = F_paths[:F_test_num]
    F_train_paths = F_paths[F_test_num:]
    M_test_paths = M_paths[:M_test_num]
    M_train_paths = M_paths[M_test_num:]

    with open(os.path.join(data_dir, "F_target_test.txt"), "w") as f:
        for pth in F_test_paths:
            f.write(pth)
            f.write("\n")

    with open(os.path.join(data_dir, "F_target_train.txt"), "w") as f:
        for pth in F_train_paths:
            f.write(pth)
            f.write("\n")

    with open(os.path.join(data_dir, "M_target_test.txt"), "w") as f:
        for pth in M_test_paths:
            f.write(pth)
            f.write("\n")

    with open(os.path.join(data_dir, "M_target_train.txt"), "w") as f:
        for pth in M_train_paths:
            f.write(pth)
            f.write("\n")


if __name__ == "__main__":
    process("/home/hhchung/data/faces_aligned_small_mirrored_co_aligned_cropped_cleaned", [2000, 2014])