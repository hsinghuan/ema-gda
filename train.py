import argparse
import os
import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
from utils import get_device, set_random_seeds, eval
import dataset
from adapter import *
from model import TwoLayerCNN, ThreeLayerCNN, TwoLayerMLPHead, Model

get_dataloader = {"rotate-mnist": dataset.get_rotate_mnist,
                  "portraits": dataset.get_portraits}

get_domain = {"rotate-mnist": dataset.rotate_mnist_domains,
              "portraits": dataset.portraits_domains}

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
        total_loss += loss.item() * data.shape[0]
        total_num += data.shape[0]

    return total_loss / total_num, total_correct / total_num

# @torch.no_grad()
# def calc_confidence(loader, encoder, head, device):
#     # find the quantile
#     total_prob = []
#     for data, _ in loader:
#         data = data.to(device)
#         logits = head(encoder(data))
#         prob = torch.softmax(logits, dim=1)
#         total_prob.append(prob)
#     total_prob = torch.cat(total_prob)
#     print(f"average max prob: {torch.mean(torch.amax(total_prob, dim=1))}")
#     print(f"average entropy: {torch.mean(torch.sum(total_prob * torch.log(total_prob), dim=1))}")
#

def source_train(args, device="cpu"):
    if args.dataset == "rotate-mnist":
        train_loader, val_loader = get_dataloader["rotate-mnist"](args.data_dir, 0, batch_size = 256, val = True)
        feat_dim = 9216
        encoder, head = TwoLayerCNN(), TwoLayerMLPHead(feat_dim, feat_dim // 2, 10)
    elif args.dataset == "portraits":
        train_loader, val_loader = get_dataloader["portraits"](args.data_dir, 0, batch_size = 256, val = True)
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

    return best_encoder, best_head, train_loader, val_loader


def main(args):
    set_random_seeds(args.random_seed)
    device = get_device(args.gpuID)
    encoder, head, src_train_loader, src_val_loader = source_train(args, device)
    print("Calculate Source Confidence")
    # calc_confidence(src_val_loader, encoder, head, device)
    if args.method == "wo-adapt":
        pass
    elif args.method == "direct-adapt":
        adapter = SelfTrainer(encoder, head, device)
        domains = get_domain[args.dataset]
        tgt_train_loader = get_dataloader[args.dataset](args.data_dir, len(domains) - 1, batch_size=256, val=False)
        confidence_q_list = [0.1]
        d_name = str(len(domains) - 1)
        adapter.adapt(d_name, tgt_train_loader, confidence_q_list, args)
        encoder, head = adapter.get_encoder_head()
    elif args.method == "gradual-selftrain":
        adapter = SelfTrainer(encoder, head, device)
        domains = get_domain[args.dataset]
        confidence_q_list = [0.1]
        for domain_idx in range(1, len(domains)):
            print(f"Domain Idx: {domain_idx}")
            d_name = str(domain_idx)
            train_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=False)
            adapter.adapt(d_name, train_loader, confidence_q_list, args)
        encoder, head = adapter.get_encoder_head()
    elif args.method == "uat":
        model = Model(encoder, head).to(device)
        adapter = UncertaintyAggregatedTeacher(model, device)
        domains = get_domain[args.dataset]
        confidence_q_list = [0.1]
        for domain_idx in range(1, len(domains)):
            print(f"Domain Idx: {domain_idx}")
            d_name = str(domain_idx)
            train_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=False)
            adapter.adapt(d_name, train_loader, confidence_q_list, args)
        model = adapter.get_model()
        encoder, head = model.get_encoder_head()
    elif args.method == "pseudo-label":
        model = Model(encoder, head).to(device)
        adapter = PseudoLabelTrainer(model, src_train_loader, src_val_loader, device)
        domains = get_domain[args.dataset]
        confidence_q_list = [0.1]
        tradeoff_list = [0.5, 1, 5]
        for domain_idx in range(1, len(domains)):
            print(f"Domain Idx: {domain_idx}")
            d_name = str(domain_idx)
            train_loader, val_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=True)
            adapter.adapt(d_name, train_loader, val_loader, confidence_q_list, tradeoff_list, args)
        model = adapter.get_model()
        encoder, head = model.get_encoder_head()
    elif args.method == "two-teachers-ens":
        model = Model(encoder, head).to(device)
        adapter = TwoTeachersEnsemble(model, device)
        domains = get_domain[args.dataset]
        confidence_q_list = [0.1]
        for domain_idx in range(1, len(domains)):
            print(f"Domain Idx: {domain_idx}")
            d_name = str(domain_idx)
            train_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=False)
            adapter.adapt(d_name, train_loader, confidence_q_list, args)
        model = adapter.get_model()
        encoder, head = model.get_encoder_head()
    elif args.method == "two-teachers-agr":
        model = Model(encoder, head).to(device)
        adapter = TwoTeachersAgreement(model, device)
        domains = get_domain[args.dataset]
        confidence_q_list = [0.1]
        for domain_idx in range(1, len(domains)):
            print(f"Domain Idx: {domain_idx}")
            d_name = str(domain_idx)
            train_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=False)
            adapter.adapt(d_name, train_loader, confidence_q_list, args)
        model = adapter.get_model()
        encoder, head = model.get_encoder_head()
    elif args.method == "uncertainty-aware-ens":
        model = Model(encoder, head).to(device)
        adapter = UncertaintyAwareEnsemble(model, device)
        domains = get_domain[args.dataset]
        confidence_q_list = [0.1]
        sharpness_list = [0.01, 0.1, 1]
        for domain_idx in range(1, len(domains)):
            print(f"Domain Idx: {domain_idx}")
            d_name = str(domain_idx)
            train_loader, val_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=True)
            adapter.adapt(d_name, train_loader, confidence_q_list, sharpness_list, args, val_loader)
        model = adapter.get_model()
        encoder, head = model.get_encoder_head()
    elif args.method == "uncertainty-plinear-ens":
        model = Model(encoder, head).to(device)
        adapter = UncertaintyPLinearEnsemble(model, device)
        domains = get_domain[args.dataset]
        confidence_q_list = [0.1]
        slope_list = [0.5, 1, 2, 4]
        for domain_idx in range(1, len(domains)):
            print(f"Domain Idx: {domain_idx}")
            d_name = str(domain_idx)
            train_loader, val_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=True)
            adapter.adapt(d_name, train_loader, confidence_q_list, slope_list, args, val_loader)
        model = adapter.get_model()
        encoder, head = model.get_encoder_head()
    elif args.method == "entropy-plinear-ens":
        model = Model(encoder, head).to(device)
        adapter = EntropyPLinearEnsemble(model, device)
        domains = get_domain[args.dataset]
        confidence_q_list = [0.1]
        if args.dataset == "rotate-mnist":
            class_num = 10
        elif args.dataset == "portraits":
            class_num = 2
        min_slope = 1 / (2 * np.log(class_num))
        slope_list = [min_slope, 2 * min_slope, 4 * min_slope]
        for domain_idx in range(1, len(domains)):
            print(f"Domain Idx: {domain_idx}")
            d_name = str(domain_idx)
            train_loader, val_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=True)
            adapter.adapt(d_name, train_loader, confidence_q_list, slope_list, args, val_loader)
        model = adapter.get_model()
        encoder, head = model.get_encoder_head()
    elif args.method == "hierarchical-teacher":
        model = Model(encoder, head).to(device)
        adapter = HierarchicalTeacher(model, device)
        domains = get_domain[args.dataset]
        confidence_q_list = [0.1]
        if args.dataset == "rotate-mnist":
            class_num = 10
        elif args.dataset == "portraits":
            class_num = 2
        min_slope = 1 / (2 * np.log(class_num))
        slope_list = [min_slope, 50 * min_slope, 100 * min_slope]
        lambda_list = [5]
        for domain_idx in range(1, len(domains)):
            print(f"Domain Idx: {domain_idx}")
            d_name = str(domain_idx)
            train_loader, val_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=True)
            adapter.adapt(d_name, train_loader, confidence_q_list, slope_list, lambda_list, args, val_loader)
        model = adapter.get_model()
        encoder, head = model.get_encoder_head()
    elif args.method == "hierarchical-teacher-sigmoid":
        model = Model(encoder, head).to(device)
        adapter = HierarchicalTeacherSigmoid(model, device)
        domains = get_domain[args.dataset]
        confidence_q_list = [0.1]
        sharpness_list = [1e-4, 1e-2, 1]
        lambda_list = [5]
        for domain_idx in range(1, len(domains)):
            print(f"Domain Idx: {domain_idx}")
            d_name = str(domain_idx)
            train_loader, val_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=True)
            adapter.adapt(d_name, train_loader, confidence_q_list, sharpness_list, lambda_list, args, val_loader)
        model = adapter.get_model()
        encoder, head = model.get_encoder_head()
    elif args.method == "entropy-plinear-calibrated-ens":
        model = Model(encoder, head).to(device)
        adapter = EntropyPLinearCalibratedEnsemble(model, device)
        domains = get_domain[args.dataset]
        confidence_q_list = [0.1]
        if args.dataset == "rotate-mnist":
            class_num = 10
        elif args.dataset == "portraits":
            class_num = 2
        min_slope = 1 / (2 * np.log(class_num))
        slope_list = [min_slope, 2 * min_slope, 3 * min_slope]
        for domain_idx in range(1, len(domains)):
            print(f"Domain Idx: {domain_idx}")
            d_name = str(domain_idx)
            train_loader, val_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=True)
            adapter.adapt(d_name, train_loader, confidence_q_list, slope_list, args, val_loader)
        model = adapter.get_model()
        encoder, head = model.get_encoder_head()
    elif args.method == "two-teachers-performances":
        model = Model(encoder, head).to(device)
        adapter = TwoTeachersPerformance(model, src_val_loader, device)
        domains = get_domain[args.dataset]
        confidence_q_list = [0.1]
        for domain_idx in range(1, len(domains)):
            print(f"Domain Idx: {domain_idx}")
            d_name = str(domain_idx)
            train_loader= get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=False)
            adapter.adapt(d_name, train_loader, confidence_q_list, args)
        model = adapter.get_model()
        encoder, head = model.get_encoder_head()



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
    parser.add_argument("--train_epochs", type=int, help="number of training epochs", default=50)
    parser.add_argument("--adapt_epochs", type=int, help="number of adaptation epochs", default=10)
    parser.add_argument("--adapt_lr", type=float, help="learning rate for adaptation optimizer", default=1e-3)
    parser.add_argument("--analyze_feat", help="whether save features", nargs='?', type=bool, const=1, default=0)
    parser.add_argument("--random_seed", type=int, help="random seed", default=42)
    parser.add_argument("--gpuID", type=int, help="which gpu to use", default=0)
    args = parser.parse_args()
    main(args)
