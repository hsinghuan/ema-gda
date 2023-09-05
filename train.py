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
    # summary(encoder)
    # summary(head)
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
        momentum_list = [0.0, 0.1, 0.2, 0.3]
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
    # elif args.method == "dagde":
    #     domains = get_domain[args.dataset]
    #     total_train_num = get_total_train_num[args.dataset]
    #     class_num = get_class_num[args.dataset]
    #     Z = torch.zeros(total_train_num, class_num, dtype=torch.float)
    #     z = torch.zeros(total_train_num, class_num, dtype=torch.float)
    #     trainloader_list = []
    #     # domain2trainloader = OrderedDict()
    #     for domain_idx in range(len(domains)):
    #         print(f"Domain Idx: {domain_idx}")
    #         if domain_idx == len(domains) - 1:
    #             train_loader, val_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256,
    #                                                                     val=True, indexed=True)
    #         else:
    #             train_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=False,
    #                                                         indexed=True)
    #         # domain2trainloader[domain_idx] = train_loader
    #         trainloader_list.append(train_loader)
    #     model = Model(encoder, head).to(device)
    #     # measure OT distance using source model
    #     dist_list = []
    #     import ot
    #     for i in range(1, len(domains)):
    #         print(f"Domain Idx: {domain_idx}")
    #         loader_i_1 = trainloader_list[domain_idx-1] # domain2trainloader[domain_idx-1]
    #         loader_i = trainloader_list[domain_idx] # domain2trainloader[domain_idx]
    #         x_i_1, x_i = [], []
    #         for _, data, _ in loader_i_1:
    #             data = data.to(device)
    #             x_i_1.append(model.feature(data))
    #
    #         for _, data, _ in loader_i:
    #             data = data.to(device)
    #             x_i.append(model.feature(data))
    #         x_i_1 = torch.cat(x_i_1)
    #         x_i = torch.cat(x_i)
    #         print("x_{i-1} shape:", x_i_1.shape, "x_{i} shape:", x_i.shape)
    #         # dist = self.dist(x_a, x_b)
    #         dist_mat = ot.dist(x_i_1, x_i).cpu().detach().numpy()
    #         n_i_1, n_i = x_i_1.shape[0], x_i.shape[0]
    #         a, b = np.ones(n_i_1) / n_i_1, np.ones(n_i) / n_i
    #         dist = ot.emd2(a, b, dist_mat)
    #         dist_list.append(dist)
    #     max_dist = max(dist_list)
    #     print(dist_list)
    #     norm_dist_list = [dist / max_dist for dist in dist_list]
    #     print(norm_dist_list)
    #     beta_list = [np.log(1/1e-15), np.log(1/0.1), np.log(1/0.2), np.log(1/0.3)]
    #     confidence_q_list = [0.1]
    #     performance_dict = dict()
    #     for beta in beta_list:  # global hyper-parameter, same across all domains
    #         # gradual adaptation
    #         adapter = DistanceAwareGradualDomainEnsemble(deepcopy(model), Z, z, beta, trainloader_list, norm_dist_list, device)
    #         for domain_idx in range(1, len(domains)):
    #             print(f"Domain Idx: {domain_idx}")
    #             adapter.adapt(domain_idx, confidence_q_list, args)
    #
    #         score = adapter.target_validate(val_loader)
    #         adapted_model = adapter.get_model()
    #         performance_dict[beta] = {'model': adapted_model, 'score': score, 'pl_acc_list': adapter.pl_acc_list, 'momentum_record_list': adapter.momentum_record_list}
    #     # model selection
    #     best_score = -np.inf
    #     best_beta = None
    #     best_model = None
    #     best_pl_acc_list = None
    #     best_momentum_record_list = None
    #     for beta, ckpt_dict in performance_dict.items():
    #         score = ckpt_dict['score']
    #         print(f"Beta: {beta} Score: {round(score, 3)}")
    #         if score > best_score:
    #             best_beta = beta
    #             best_model = ckpt_dict['model']
    #             best_pl_acc_list = ckpt_dict['pl_acc_list']
    #             best_momentum_record_list = ckpt_dict['momentum_record_list']
    #             best_score = score
    #     print(f"Best beta: {best_beta} Best score: {round(best_score, 3)} PL Acc List:", best_pl_acc_list)
    #     print("Normalized Distance", [round(norm_dist, 3) for norm_dist in norm_dist_list])
    #     print("Momentum Record", [round(momentum_record, 3) for momentum_record in best_momentum_record_list])
    #     model = best_model
    #     encoder, head = model.get_encoder_head()
    elif args.method == "pdagde":
        domains = get_domain[args.dataset]
        total_train_num = get_total_train_num[args.dataset]
        class_num = get_class_num[args.dataset]
        Z = torch.zeros(total_train_num, class_num, dtype=torch.float)
        z = torch.zeros(total_train_num, class_num, dtype=torch.float)
        trainloader_list = []
        for domain_idx in range(len(domains)):
            print(f"Domain Idx: {domain_idx}")
            if domain_idx == len(domains) - 1:
                train_loader, val_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256,
                                                                        val=True, indexed=True)
            else:
                train_loader = get_dataloader[args.dataset](args.data_dir, domain_idx, batch_size=256, val=False,
                                                            indexed=True)
            trainloader_list.append(train_loader)
        model = Model(encoder, head).to(device)
        # measure OT distance using source model
        dist_lists = []
        import ot
        for i in range(len(domains)-1): # from S_{0} to S_{T-1}
            d_list = []
            for j in range(i + 1, len(domains)): # from S_{i} to S_{T}
                print(f"i: {i} j: {j}")
                loader_i = trainloader_list[i]
                loader_j = trainloader_list[j]
                x_i, x_j = [], []
                for _, data, _ in loader_i:
                    data = data.to(device)
                    x_i.append(model.feature(data).detach().cpu())
                for _, data, _ in loader_j:
                    data = data.to(device)
                    x_j.append(model.feature(data).detach().cpu())
                x_i = torch.cat(x_i)
                x_j = torch.cat(x_j)
                print("x_{i} shape:", x_i.shape, "x_{j} shape:", x_j.shape)
                dist_mat = ot.dist(x_i, x_j).cpu().detach().numpy()
                n_i, n_j = x_i.shape[0], x_j.shape[0]
                a, b = np.ones(n_i) / n_i, np.ones(n_j) / n_j
                dist = ot.emd2(a, b, dist_mat)
                d_list.append(dist)
            dist_lists.append(d_list)

        beta_list = [-np.log(1e-15), -np.log(1/8), -np.log(1/4), -np.log(1/2)]
        confidence_q_list = [0.1]
        performance_dict = dict()
        for beta in beta_list:  # global hyper-parameter, same across all domains
            # gradual adaptation
            adapter = PairwiseDistanceAwareGradualDomainEnsemble(deepcopy(model), Z, z, beta, trainloader_list, dist_lists, device)
            for domain_idx in range(1, len(domains)):
                print(f"Domain Idx: {domain_idx}")
                adapter.adapt(domain_idx, confidence_q_list, args)

            score = adapter.target_validate(val_loader)
            adapted_model = adapter.get_model()
            performance_dict[beta] = {'model': adapted_model, 'score': score, 'pl_acc_list': adapter.pl_acc_list}
        # model selection
        best_score = -np.inf
        best_beta = None
        best_model = None
        best_pl_acc_list = None
        for beta, ckpt_dict in performance_dict.items():
            score = ckpt_dict['score']
            print(f"Beta: {beta} Score: {round(score, 3)}")
            if score > best_score:
                best_beta = beta
                best_model = ckpt_dict['model']
                best_pl_acc_list = ckpt_dict['pl_acc_list']
                best_score = score
        print(f"Best beta: {best_beta} Best score: {round(best_score, 3)} PL Acc List:", best_pl_acc_list)
        model = best_model
        encoder, head = model.get_encoder_head()
        # for dlist in dist_lists:
        #     print(dlist)

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
    parser.add_argument("--adapt_epochs", type=int, help="number of adaptation epochs", default=20)
    parser.add_argument("--adapt_lr", type=float, help="learning rate for adaptation optimizer", default=1e-3)
    parser.add_argument("--analyze_feat", help="whether save features", nargs='?', type=bool, const=1, default=0)
    parser.add_argument("--random_seed", type=int, help="random seed", default=42)
    parser.add_argument("--gpuID", type=int, help="which gpu to use", default=0)
    args = parser.parse_args()
    main(args)
