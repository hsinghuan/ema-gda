import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pytorch_adapt.validators import IMValidator


class GradualDomainEnsemble:
    def __init__(self, model, Z, z, momentum, device="cpu"):
        self.device = device
        self.model = model.to(self.device)
        self.Z = Z.to(self.device) # intermediate values
        self.z = z.to(self.device)# temporal outputs
        self.momentum = momentum
        self.validator = IMValidator()
        self.pl_acc_list = []

    def _adapt_train_epoch(self, model, train_loader, optimizer, alpha):
        model.train()
        total_loss = 0
        total_num = 0
        total_logits = []
        for idx, data, _ in train_loader:
            data = data.to(self.device)
            student_logits = model(data)
            ensemble_probs = self.z[idx]
            loss, mask = self._pseudo_label_loss(student_logits, ensemble_probs, alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mask.sum().item()
            total_num += mask.sum().item()
            total_logits.append(student_logits)
        total_loss /= total_num
        total_logits = torch.cat(total_logits)
        score = self.validator(target_train={"logits": total_logits})
        return total_loss, score

    def _adapt_eval_epoch(self, model, val_loader, alpha):
        model.eval()
        total_loss = 0
        total_num = 0
        total_logits = []
        for idx, data, _ in val_loader:
            data = data.to(self.device)
            student_logits = model(data)
            ensemble_probs = self.z[idx]
            loss, mask = self._pseudo_label_loss(student_logits, ensemble_probs, alpha)
            total_loss += loss.item() * mask.sum().item()
            total_num += mask.sum().item()
            total_logits.append(student_logits)
        total_loss /= total_num
        total_logits = torch.cat(total_logits)
        score = self.validator(target_train={"logits": total_logits})
        return total_loss, score

    @torch.no_grad()
    def _oracle_eval_epoch(self, model, val_loader):
        model.eval()
        total_correct = 0
        total_pl_correct = 0
        total_num = 0
        for idx, data, y in val_loader:
            data, y = data.to(self.device), y.to(self.device)
            output = model(data)

            pred = torch.argmax(output, dim=1)
            total_correct += torch.eq(pred, y).sum().item()

            pl = torch.argmax(self.z[idx], dim=1)
            total_pl_correct += torch.eq(pl, y).sum().item()
            total_num += data.shape[0]

        return total_correct / total_num, total_pl_correct / total_num


    def _adapt_train_eval(self, domain_idx, domain2trainloader, confidence_q, args):
        # update Z first (given that Z is initialized to 0 and source model has been trained)
        self._update_Z(domain2trainloader, domain_idx)
        train_loader = domain2trainloader[domain_idx]
        alpha = self._calc_alpha(train_loader, confidence_q) # calculate from Z (accumulated prediction)

        model = deepcopy(self.model).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.adapt_lr)
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_score = self._adapt_train_epoch(model, train_loader, optimizer, alpha)
            train_acc, pl_acc = self._oracle_eval_epoch(model, train_loader)

            print(f"Momentum: {self.momentum} Confidence q: {confidence_q} Epoch: {e} Train Loss: {train_loss} Train Acc: {train_acc} PL Acc: {pl_acc}")
            self.writer.add_scalar("Loss/train", train_loss, e)
            self.writer.add_scalar("Score/train", train_score, e)

        self.pl_acc_list.append(pl_acc)
        return model, train_score


    @torch.no_grad()
    def _update_Z(self, domain2trainloader, domain_idx):
        self.model.eval()
        for d, loader in domain2trainloader.items():
            if d < domain_idx:
                continue
            for idx, img, _ in loader:
                img = img.to(self.device)
                output = self.model(img)
                probs = F.softmax(output, dim=1)
                self.Z[idx] = self.momentum * self.Z[idx] + (1 - self.momentum) * probs
                self.z[idx] = self.Z[idx] * (1. / (1. - self.momentum ** domain_idx))

    @torch.no_grad()
    def _calc_alpha(self, loader, confidence_q):
        # find the quantile
        total_prob = []
        for idx, _, _ in loader:
            total_prob.append(self.Z[idx])
        total_prob = torch.cat(total_prob)
        confidence = torch.amax(total_prob, 1) - torch.amin(total_prob, 1)
        alpha = torch.quantile(confidence, confidence_q)

        return alpha

    def _pseudo_label_loss(self, student_logits, ensemble_probs, alpha):
        confidence = torch.amax(ensemble_probs, 1) - torch.amin(ensemble_probs, 1)
        mask = confidence >= alpha
        teacher_pred = torch.argmax(ensemble_probs, dim=1)
        pseudo_loss = (F.nll_loss(F.log_softmax(student_logits, dim=1), teacher_pred, reduction='none') * mask).mean()
        return pseudo_loss, mask

    def adapt(self, domain_idx, domain2trainloader, confidence_q_list, args):
        performance_dict = dict()
        for confidence_q in confidence_q_list:
            run_name = f"{args.method}_{self.momentum}_{confidence_q}_{args.random_seed}"
            self.writer = SummaryWriter(os.path.join(args.log_dir, args.dataset, str(domain_idx), run_name))
            model, val_score = self._adapt_train_eval(domain_idx, domain2trainloader, confidence_q, args)
            performance_dict[confidence_q] = {"model": model, "score": val_score}

        best_score = -np.inf
        best_model = None
        for confidence_q, ckpt_dict in performance_dict.items():
            if ckpt_dict["score"] > best_score:
                best_model = ckpt_dict["model"]
                best_score = ckpt_dict["score"]

        self.model = deepcopy(best_model).to(self.device)

    def target_validate(self, val_loader):
        total_logits = []
        for _, img, _ in val_loader:
            img = img.to(self.device)
            logits = self.model(img)
            total_logits.append(logits)
        total_logits = torch.cat(total_logits)
        score = self.validator(target_train={"logits": total_logits})
        return score

    def get_model(self):
        return self.model

