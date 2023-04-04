import argparse
import os
from copy import deepcopy
import torch
import torch.nn.functional as F
from utils import get_device, set_random_seeds, eval
import dataset
from model import TwoLayerCNN, ThreeLayerCNN, TwoLayerMLPHead

def load_data_model(args, device="cpu"):
    if args.dataset == "rotate-mnist":
        test_loader = dataset.get_rotate_mnist(args.data_dir, len(dataset.rotate_mnist_domains) - 1, batch_size = 256, target_test=True)
        feat_dim = 9216
        encoder, head = TwoLayerCNN(), TwoLayerMLPHead(feat_dim, feat_dim // 2, 10)
    elif args.dataset == "portraits":
        test_loader = dataset.get_portraits(args.data_dir, len(dataset.portraits_domains) -1, batch_size = 256, target_test=True)
        feat_dim = 6272
        encoder, head = ThreeLayerCNN(), TwoLayerMLPHead(feat_dim, feat_dim // 2, 2)
    encoder, head = encoder.to(device), head.to(device)
    state_dict = torch.load(os.path.join(args.ckpt_dir, args.dataset, f'{args.method}_{args.random_seed}.pt'))
    encoder.load_state_dict(state_dict["encoder"])
    head.load_state_dict(state_dict["head"])
    return test_loader, encoder, head

def main(args):
    set_random_seeds(args.random_seed)
    device = get_device(args.gpuID)
    test_loader, encoder, head = load_data_model(args, device)
    _, test_acc = eval(test_loader, encoder, head, device=device)
    print(test_acc)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="name of the dataset")
    parser.add_argument("--data_dir", type=str, help="path to data directory")
    parser.add_argument("--log_dir", type=str, help="path to log directory", default="runs")
    parser.add_argument("--ckpt_dir", type=str, help="path to model checkpoint directory", default="checkpoints")
    parser.add_argument("--result_dir", type=str, help="path to performance results directory", default="results")
    parser.add_argument("--method", type=str, help="adaptation method")
    parser.add_argument("--analyze_feat", help="whether save features", nargs='?', type=bool, const=1, default=0)
    parser.add_argument("--random_seed", type=int, help="random seed", default=42)
    parser.add_argument("--gpuID", type=int, help="which gpu to use", default=0)
    args = parser.parse_args()
    main(args)