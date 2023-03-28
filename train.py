import argparse
import os
from copy import deepcopy
import torch
import torch.nn.functional as F
from utils import get_device, set_random_seeds
import dataset
from model import TwoLayerCNN, ThreeLayerCNN, TwoLayerMLPHead


def train(loader, encoder, head, optimizer, device="cpu"):
    encoder.train()
    head.train()
    total_loss = 0
    total_correct = 0
    total_num = 0
    for data, y in loader:
        data, y = data.to(device), y.to(device)
        output = head(encoder(data))
        loss = F.nll_loss(F.log_softmax(output, dim=1), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = torch.argmax(output, dim=1)
        total_correct += torch.eq(pred, y).sum().item()
        total_loss += loss.item()
        total_num += data.shape[0]

    return total_loss / total_num, total_correct / total_num

@torch.no_grad()
def eval(loader, encoder, head, device="cpu"):
    encoder.eval()
    head.eval()
    total_loss = 0
    total_correct = 0
    total_num = 0
    for data, y in loader:
        data, y = data.to(device), y.to(device)
        output = head(encoder(data))
        loss = F.nll_loss(F.log_softmax(output, dim=1), y)

        pred = torch.argmax(output, dim=1)
        total_correct += torch.eq(pred, y).sum().item()
        total_loss += loss.item()
        total_num += data.shape[0]

    return total_loss / total_num, total_correct / total_num



def source_train(args, device="cpu"):
    if args.dataset == "rotate-mnist":
        train_loader, val_loader = dataset.get_rotate_mnist(args.data_dir, dataset.rotate_mnist_domains[0], batch_size = 256, val = True)
        feat_dim = 9216
        encoder, head = TwoLayerCNN(), TwoLayerMLPHead(feat_dim, feat_dim // 2, 10)
    elif args.dataset == "portraits":
        train_loader, val_loader = dataset.get_portraits(args.data_dir, dataset.portraits_domains[0], batch_size = 256, val = True)
        feat_dim = 6272
        encoder, head = ThreeLayerCNN(), TwoLayerMLPHead(feat_dim, feat_dim // 2, 2)

    encoder, head = encoder.to(device), head.to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-3)

    best_val_acc = 0
    best_encoder, best_head = None, None
    staleness = 0
    patience = 5

    for e in range(1, args.train_epochs + 1):
        train_loss, train_acc = train(train_loader, encoder, head, optimizer, device=device)
        val_loss, val_acc = eval(val_loader, encoder, head, device=device)
        print(f"Epoch: {e} Train Loss: {round(train_loss, 3)} Train Acc: {round(train_acc, 3)} Val Loss: {round(val_loss, 3)} Val Acc: {round(val_acc, 3)}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_encoder, best_head = deepcopy(encoder), deepcopy(head)
            staleness = 0
        else:
            staleness += 1

        if staleness > patience:
            break

    return best_encoder, best_head


def main(args):
    set_random_seeds(args.random_seed)
    device = get_device(args.gpuID)
    encoder, head = source_train(args, device)
    if args.method == "wo-adapt":
        pass

    # save encoder, head
    os.makedirs(os.path.join(args.ckpt_dir, args.dataset), exist_ok=True)
    torch.save({"encoder": encoder.state_dict(),
                "head": head.state_dict()},
               os.path.join(args.ckpt_dir, args.dataset, f'{args.method}_{args.random_seed}.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="name of the dataset")
    parser.add_argument("--data_dir", type=str, help="path to data directory")
    parser.add_argument("--log_dir", type=str, help="path to log directory", default="runs")
    parser.add_argument("--ckpt_dir", type=str, help="path to model checkpoint directory", default="checkpoints")
    parser.add_argument("--result_dir", type=str, help="path to performance results directory", default="results")
    parser.add_argument("--method", type=str, help="adaptation method")
    parser.add_argument("--train_epochs", type=int, help="number of training epochs", default=100)
    parser.add_argument("--adapt_epochs", type=int, help="number of adaptation epochs", default=100)
    parser.add_argument("--adapt_lr", type=float, help="learning rate for adaptation optimizer", default=1e-3)
    parser.add_argument("--analyze_feat", help="whether save features", nargs='?', type=bool, const=1, default=0)
    parser.add_argument("--random_seed", type=int, help="random seed", default=42)
    parser.add_argument("--gpuID", type=int, help="which gpu to use", default=0)
    args = parser.parse_args()
    main(args)