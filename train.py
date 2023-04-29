import argparse
import os
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torchsummary import summary
from utils import get_device, set_random_seeds, eval
import dataset
from adapter import *
from model import TwoLayerCNN, ThreeLayerCNN, TwoLayerMLPHead, Model

get_dataloader = {"rotate-mnist": dataset.get_rotate_mnist,
                  "portraits": dataset.get_portraits}

get_domain = {"rotate-mnist": dataset.rotate_mnist_domains,
              "portraits": dataset.portraits_domains}

get_total_train_num = {"rotate-mnist": dataset.rotate_mnist_total_train_num,
                       "portraits": dataset.portraits_total_train_num}

get_class_num = {"rotate-mnist": dataset.rotate_mnist_class_num,
                 "portraits": dataset.portraits_class_num}

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
    summary(encoder)
    summary(head)
    # print("Calculate Source Confidence")
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
        print("PL Acc List:", adapter.pl_acc_list)
    # elif args.method == "uat":
    #     model = Model(encoder, head).to(device)
    #     adapter = UncertaintyAggregatedTeacher(model, device)
    #     domains = get_domain[args.dataset]
    #     confidence_q_list = [0.1]
    #     for domain_idx in range(1, len(domains)):
    #         print(f"Domain Idx: {domain_idx}")
    #         d_name = str(domain_idx)
    #         train_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=False)
    #         adapter.adapt(d_name, train_loader, confidence_q_list, args)
    #     model = adapter.get_model()
    #     encoder, head = model.get_encoder_head()
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
    elif args.method == "gradual-domain-ensemble":
        domains = get_domain[args.dataset]
        total_train_num = get_total_train_num[args.dataset]
        class_num = get_class_num[args.dataset]
        Z = torch.zeros(total_train_num, class_num, dtype=torch.float)
        z = torch.zeros(total_train_num, class_num, dtype=torch.float)
        domain2trainloader = OrderedDict()
        for domain_idx in range(1, len(domains)):
            print(f"Domain Idx: {domain_idx}")
            if domain_idx == len(domains) - 1:
                train_loader, val_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=True, indexed=True)
            else:
                train_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=False, indexed=True)
            domain2trainloader[domain_idx] = train_loader

        model = Model(encoder, head).to(device)
        momentum_list = [0.5] # [0.1, 0.3, 0.5]
        confidence_q_list = [0.1]
        performance_dict = dict()
        for momentum in momentum_list: # global hyper-parameter, same across all domains
            # gradual adaptation
            adapter = GradualDomainEnsemble(deepcopy(model), Z, z, momentum, device)
            for domain_idx in range(1, len(domains)):
                print(f"Domain Idx: {domain_idx}")
                adapter.adapt(domain_idx, domain2trainloader, confidence_q_list, args)

            score = adapter.target_validate(val_loader)
            adapted_model = adapter.get_model()
            performance_dict[momentum] = {'model': adapted_model, 'score': score, 'pl_acc_list': adapter.pl_acc_list}
        # hyper-parameter selection
        best_score = -np.inf
        best_momentum = None
        best_model = None
        best_pl_acc_list = None
        for momentum, ckpt_dict in performance_dict.items():
            score = ckpt_dict['score']
            print(f"Momentum: {momentum} Score: {round(score, 3)}")
            if score > best_score:
                best_momentum = momentum
                best_model = ckpt_dict['model']
                best_pl_acc_list = ckpt_dict['pl_acc_list']
                best_score = score
        print(f"Best momentum: {best_momentum} Best score: {round(best_score, 3)} Best PL Acc List: {best_pl_acc_list}")
        model = best_model
        encoder, head = model.get_encoder_head()

    elif args.method == "uagde":
        domains = get_domain[args.dataset]
        total_train_num = get_total_train_num[args.dataset]
        class_num = get_class_num[args.dataset]
        # Z = torch.ones(total_train_num, class_num, dtype=torch.float) / class_num
        # z = torch.ones(total_train_num, class_num, dtype=torch.float) / class_num
        Z = torch.zeros(total_train_num, class_num, dtype=torch.float)
        z = torch.zeros(total_train_num, class_num, dtype=torch.float)
        domain2trainloader = OrderedDict()
        for domain_idx in range(1, len(domains)):
            print(f"Domain Idx: {domain_idx}")
            if domain_idx == len(domains) - 1:
                train_loader, val_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=True, indexed=True)
            else:
                train_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=False, indexed=True)
            domain2trainloader[domain_idx] = train_loader

        model = Model(encoder, head).to(device)
        min_slope = 1 / (2 * np.log(class_num))
        slope_list = [50 * min_slope] # [10 * min_slope, 30 * min_slope, 50 * min_slope]
        # mid_slope = np.log(2)
        # slope_list = [1/4 * mid_slope]
        confidence_q_list = [0.1]
        performance_dict = dict()
        for slope in slope_list: # global hyper-parameter, same across all domains
            # gradual adaptation
            adapter = UncertaintyAwareGradualDomainEnsemble(deepcopy(model), Z, z, slope, device)
            for domain_idx in range(1, len(domains)):
                print(f"Domain Idx: {domain_idx}")
                adapter.adapt(domain_idx, domain2trainloader, confidence_q_list, args)

            score = adapter.target_validate(val_loader)
            adapted_model = adapter.get_model()
            performance_dict[slope] = {'model': adapted_model, 'score': score, 'pl_acc_list': adapter.pl_acc_list}
        # model selection
        best_score = -np.inf
        best_slope = None
        best_model = None
        best_pl_acc_list = None
        for slope, ckpt_dict in performance_dict.items():
            score = ckpt_dict['score']
            print(f"Slope: {slope} Score: {round(score, 3)}")
            if score > best_score:
                best_slope = slope
                best_model = ckpt_dict['model']
                best_pl_acc_list = ckpt_dict['pl_acc_list']
                best_score = score
        print(f"Best slope: {best_slope} Best score: {round(best_score, 3)} PL Acc List:", best_pl_acc_list)
        model = best_model
        encoder, head = model.get_encoder_head()
    elif args.method == "dagde":
        domains = get_domain[args.dataset]
        total_train_num = get_total_train_num[args.dataset]
        class_num = get_class_num[args.dataset]
        Z = torch.zeros(total_train_num, class_num, dtype=torch.float)
        z = torch.zeros(total_train_num, class_num, dtype=torch.float)
        domain2trainloader = OrderedDict()
        for domain_idx in range(len(domains)):
            print(f"Domain Idx: {domain_idx}")
            if domain_idx == len(domains) - 1:
                train_loader, val_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256,
                                                                        val=True, indexed=True)
            else:
                train_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=False,
                                                            indexed=True)
            domain2trainloader[domain_idx] = train_loader

        model = Model(encoder, head).to(device)
        beta_list = [2e-3]
        confidence_q_list = [0.1]
        performance_dict = dict()
        for beta in beta_list:  # global hyper-parameter, same across all domains
            # gradual adaptation
            adapter = DistanceAwareGradualDomainEnsemble(deepcopy(model), Z, z, beta, device)
            for domain_idx in range(1, len(domains)):
                print(f"Domain Idx: {domain_idx}")
                adapter.adapt(domain_idx, domain2trainloader, confidence_q_list, args)

            score = adapter.target_validate(val_loader)
            adapted_model = adapter.get_model()
            performance_dict[beta] = {'model': adapted_model, 'score': score, 'pl_acc_list': adapter.pl_acc_list}
        # model selection
        best_score = -np.inf
        best_slope = None
        best_model = None
        best_pl_acc_list = None
        for slope, ckpt_dict in performance_dict.items():
            score = ckpt_dict['score']
            print(f"Slope: {slope} Score: {round(score, 3)}")
            if score > best_score:
                best_slope = slope
                best_model = ckpt_dict['model']
                best_pl_acc_list = ckpt_dict['pl_acc_list']
                best_score = score
        print(f"Best slope: {best_slope} Best score: {round(best_score, 3)} PL Acc List:", best_pl_acc_list)
        model = best_model
        encoder, head = model.get_encoder_head()

    # elif args.method == "uncertainty-aware-ens":
    #     model = Model(encoder, head).to(device)
    #     adapter = UncertaintyAwareEnsemble(model, device)
    #     domains = get_domain[args.dataset]
    #     confidence_q_list = [0.1]
    #     sharpness_list = [0.01, 0.1, 1]
    #     for domain_idx in range(1, len(domains)):
    #         print(f"Domain Idx: {domain_idx}")
    #         d_name = str(domain_idx)
    #         train_loader, val_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=True)
    #         adapter.adapt(d_name, train_loader, confidence_q_list, sharpness_list, args, val_loader)
    #     model = adapter.get_model()
    #     encoder, head = model.get_encoder_head()
    # elif args.method == "uncertainty-plinear-ens":
    #     model = Model(encoder, head).to(device)
    #     adapter = UncertaintyPLinearEnsemble(model, device)
    #     domains = get_domain[args.dataset]
    #     confidence_q_list = [0.1]
    #     slope_list = [0.5, 1, 2, 4]
    #     for domain_idx in range(1, len(domains)):
    #         print(f"Domain Idx: {domain_idx}")
    #         d_name = str(domain_idx)
    #         train_loader, val_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=True)
    #         adapter.adapt(d_name, train_loader, confidence_q_list, slope_list, args, val_loader)
    #     model = adapter.get_model()
    #     encoder, head = model.get_encoder_head()
    # elif args.method == "entropy-plinear-ens":
    #     model = Model(encoder, head).to(device)
    #     adapter = EntropyPLinearEnsemble(model, device)
    #     domains = get_domain[args.dataset]
    #     confidence_q_list = [0.1]
    #     if args.dataset == "rotate-mnist":
    #         class_num = 10
    #     elif args.dataset == "portraits":
    #         class_num = 2
    #     min_slope = 1 / (2 * np.log(class_num))
    #     slope_list = [min_slope, 2 * min_slope, 4 * min_slope]
    #     for domain_idx in range(1, len(domains)):
    #         print(f"Domain Idx: {domain_idx}")
    #         d_name = str(domain_idx)
    #         train_loader, val_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=True)
    #         adapter.adapt(d_name, train_loader, confidence_q_list, slope_list, args, val_loader)
    #     model = adapter.get_model()
    #     encoder, head = model.get_encoder_head()
    # elif args.method == "entropy-sigmoid-ens":
    #     model = Model(encoder, head).to(device)
    #     adapter = EntropySigmoidEnsemble(model, device)
    #     domains = get_domain[args.dataset]
    #     confidence_q_list = [0.1]
    #     sharpness_list = [2**-4, 2**-2, 2**0]
    #     for domain_idx in range(1, len(domains)):
    #         print(f"Domain Idx: {domain_idx}")
    #         d_name = str(domain_idx)
    #         train_loader, val_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=True)
    #         adapter.adapt(d_name, train_loader, confidence_q_list, sharpness_list, args, val_loader)
    #     model = adapter.get_model()
    #     encoder, head = model.get_encoder_head()
    # elif args.method == "hierarchical-teacher":
    #     model = Model(encoder, head).to(device)
    #     adapter = HierarchicalTeacher(model, device)
    #     domains = get_domain[args.dataset]
    #     confidence_q_list = [0.1]
    #     if args.dataset == "rotate-mnist":
    #         class_num = 10
    #     elif args.dataset == "portraits":
    #         class_num = 2
    #     min_slope = 1 / (2 * np.log(class_num))
    #     slope_list = [min_slope, 50 * min_slope, 100 * min_slope]
    #     lambda_list = [5]
    #     for domain_idx in range(1, len(domains)):
    #         print(f"Domain Idx: {domain_idx}")
    #         d_name = str(domain_idx)
    #         train_loader, val_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=True)
    #         adapter.adapt(d_name, train_loader, confidence_q_list, slope_list, lambda_list, args, val_loader)
    #     model = adapter.get_model()
    #     encoder, head = model.get_encoder_head()
    # elif args.method == "hierarchical-teacher-sigmoid":
    #     model = Model(encoder, head).to(device)
    #     adapter = HierarchicalTeacherSigmoid(model, device)
    #     domains = get_domain[args.dataset]
    #     confidence_q_list = [0.1]
    #     sharpness_list = [1e-4, 1e-2, 1]
    #     lambda_list = [5]
    #     for domain_idx in range(1, len(domains)):
    #         print(f"Domain Idx: {domain_idx}")
    #         d_name = str(domain_idx)
    #         train_loader, val_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=True)
    #         adapter.adapt(d_name, train_loader, confidence_q_list, sharpness_list, lambda_list, args, val_loader)
    #     model = adapter.get_model()
    #     encoder, head = model.get_encoder_head()




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
