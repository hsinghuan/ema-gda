import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_adapt.validators import SNDValidator


class UncertaintyAwareEnsemble:
    def __init__(self, model, device="cpu"):
        self.device = device
        self.model = model.to(self.device)
        self.src_teacher = deepcopy(model).to(self.device)
        self.validator = SNDValidator()

    def _adapt_train_epoch(self, model, train_loader, optimizer, alpha, src_weight):
        model.train()
        total_loss = 0
        total_num = 0
        total_logits = []
        for data, _ in train_loader:
            data = data.to(self.device)
            student_logits = model(data)
            cur_teacher_logits = self.model(data)
            src_teacher_logits = self.src_teacher(data)
            teacher_ensemble = src_weight * src_teacher_logits + (1 - src_weight) * cur_teacher_logits
            loss, mask, _ = self._pseudo_label_loss(student_logits, teacher_ensemble, alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mask.sum().item()
            total_num += mask.sum().item()
            total_logits.append(student_logits)
        total_loss /= total_num
        total_logits = torch.cat(total_logits)
        score = self.validator(target_train={"preds": total_logits})
        return total_loss, score


    @torch.no_grad()
    def _adapt_eval_epoch(self, model, val_loader, alpha, src_weight):
        model.eval()
        total_loss = 0
        total_num = 0
        total_logits = []
        for data, _ in val_loader:
            data = data.to(self.device)
            student_logits = model(data)
            cur_teacher_logits = self.model(data)
            src_teacher_logits = self.src_teacher(data)
            teacher_ensemble = src_weight * src_teacher_logits + (1 - src_weight) * cur_teacher_logits
            loss, mask, _ = self._pseudo_label_loss(student_logits, teacher_ensemble, alpha)
            total_loss += loss.item() * mask.sum().item()
            total_num += mask.sum().item()
            total_logits.append(student_logits)
        total_loss /= total_num
        total_logits = torch.cat(total_logits)
        score = self.validator(target_train={"preds": total_logits})
        return total_loss, score

    @torch.no_grad()
    def _oracle_eval_epoch(self, model, val_loader):
        model.eval()
        total_correct = 0
        total_num = 0
        for data, y in val_loader:
            data, y = data.to(self.device), y.to(self.device)
            output = model(data)

            pred = torch.argmax(output, dim=1)
            total_correct += torch.eq(pred, y).sum().item()
            total_num += data.shape[0]

        return total_correct / total_num

    def _adapt_train_eval(self, train_loader, confidence_q, sharpness, args, val_loader=None):
        src_weight = self._calc_src_weight(train_loader, sharpness)
        alpha = self._calc_alpha(train_loader, confidence_q, src_weight)
        model = deepcopy(self.model).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.adapt_lr)
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_score = self._adapt_train_epoch(model, train_loader, optimizer, alpha, src_weight)
            train_acc = self._oracle_eval_epoch(model, train_loader)
            if not val_loader:
                score = train_score
                print(f"Confidence q: {confidence_q} Sharpness: {sharpness} Epoch: {e} Train Loss: {round(train_loss, 3)} Train Acc: {round(train_acc, 3)}")
                continue

            val_loss, val_score = self._adapt_eval_epoch(model, val_loader, alpha, src_weight)
            val_acc = self._oracle_eval_epoch(model, val_loader)
            score = val_score
            print(f"Confidence q: {confidence_q} Sharpness: {sharpness} Epoch: {e} Train Loss: {round(train_loss, 3)} Train Acc: {round(train_acc, 3)} Val Loss: {round(val_loss, 3)} Val Acc: {round(val_acc, 3)}")

            self.writer.add_scalar("Loss/train", train_loss, e)
            self.writer.add_scalar("Loss/val", val_loss, e)
            self.writer.add_scalar("Score/train", train_score, e)
            self.writer.add_scalar("Score/val", val_score, e)
            self.writer.add_scalar("Acc/train", train_acc, e)
            self.writer.add_scalar("Acc/val", val_acc, e)


        return model, score

    def _entropy(self, logits):
        entropies = -torch.sum(
            torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1), dim=1
        )
        return torch.mean(entropies)

    @torch.no_grad()
    def _calc_src_weight(self, loader, sharpness):
        self.src_teacher.eval()
        self.model.eval()

        src_teacher_logits = []
        cur_teacher_logits = []
        for data, _ in loader:
            data = data.to(self.device)
            src_teacher_logits.append(self.src_teacher(data).detach().cpu())
            cur_teacher_logits.append(self.model(data).detach().cpu())
        src_teacher_logits = torch.cat(src_teacher_logits)
        cur_teacher_logits = torch.cat(cur_teacher_logits)
        src_teacher_entropy = self._entropy(src_teacher_logits)
        cur_teacher_entropy = self._entropy(cur_teacher_logits)

        # src_weight = torch.exp(-torch.log(torch.tensor(2.)) * src_teacher_entropy / cur_teacher_entropy)
        src_weight = torch.sigmoid((cur_teacher_entropy - src_teacher_entropy) / sharpness)
        print(f"src teacher entropy: {round(src_teacher_entropy.item(), 4)} cur teacher entropy: {round(cur_teacher_entropy.item(), 4)} src weight: {round(src_weight.item(), 4)}")
        return src_weight

    @torch.no_grad()
    def _calc_alpha(self, loader, confidence_q, src_weight):
        # find the quantile
        total_prob = []
        # cur_teacher_pred = []
        # src_teacher_pred = []
        for data, _ in loader:
            data = data.to(self.device)
            cur_teacher_logits = self.model(data)
            src_teacher_logits = self.src_teacher(data)
            teacher_ensemble =  src_weight * src_teacher_logits + (1 - src_weight) * cur_teacher_logits
            prob = torch.softmax(teacher_ensemble, dim=1)
            total_prob.append(prob)
        total_prob = torch.cat(total_prob)
        confidence = torch.amax(total_prob, 1) - torch.amin(total_prob, 1)
        alpha = torch.quantile(confidence, confidence_q)
        return alpha

    def _pseudo_label_loss(self, y, y_target, alpha):
        prob = torch.softmax(y_target, dim=1)
        confidence = torch.amax(prob, 1) - torch.amin(prob, 1)
        mask = confidence >= alpha
        pseudo_labels = torch.argmax(prob, dim=1)
        pseudo_loss = (F.nll_loss(F.log_softmax(y, dim=1), pseudo_labels, reduction='none') * mask).mean()
        return pseudo_loss, mask, pseudo_labels



    def adapt(self, domain_name, train_loader, confidence_q_list, sharpness_list, args, val_loader=None):
        performance_dict = dict()
        for confidence_q in confidence_q_list:
            for sharpness in sharpness_list:
                run_name = f"{args.method}_{confidence_q}_{sharpness}_{args.random_seed}"
                self.writer = SummaryWriter(os.path.join(args.log_dir, args.dataset, domain_name, run_name))
                model, score = self._adapt_train_eval(train_loader, confidence_q, sharpness, args, val_loader)
                performance_dict[(confidence_q, sharpness)] = {"model": model, "score": score}


        best_score = -np.inf
        best_confidence_q = None
        best_sharpness = None
        best_model = None
        for (confidence_q, sharpness), ckpt_dict in performance_dict.items():
            score = ckpt_dict["score"]
            if score > best_score:
                best_model = ckpt_dict["model"]
                best_confidence_q = confidence_q
                best_sharpness = sharpness
                best_score = score
            print(f"Confidence Q: {confidence_q} Sharpness: {sharpness} Score: {score}")

        print(f"Best Confidence Q: {best_confidence_q} Best Sharpness: {best_sharpness}")
        # use all data to train
        if val_loader:
            # TODO: merge two dataloaders
            data_all, y_all  = [], []
            for data, y in train_loader:
                data_all.append(data)
                y_all.append(y)
            for data, y in val_loader:
                data_all.append(data)
                y_all.append(y)
            data_all = torch.cat(data_all)
            y_all = torch.cat(y_all)
            dataset = TensorDataset(data_all, y_all)
            loader = DataLoader(dataset, batch_size=256, shuffle=True)
            # TODO: rerun with the found best hyper-parameters on merged dataloader
            run_name = f"{args.method}_all_{best_confidence_q}_{best_sharpness}_{args.random_seed}"
            self.writer = SummaryWriter(os.path.join(args.log_dir, args.dataset, domain_name, run_name))
            best_model, _ = self._adapt_train_eval(loader, best_confidence_q, best_sharpness, args)


        self.model = deepcopy(best_model).to(self.device)

    def get_model(self):
        return self.model

