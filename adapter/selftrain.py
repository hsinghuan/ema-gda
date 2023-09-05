import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pytorch_adapt.validators import IMValidator


class SelfTrainer:
    def __init__(self, encoder, head, device="cpu"):
        self.device = device
        self._set_encoder_head(encoder, head)
        self.validator = IMValidator()
        self.pl_acc_list = []

    def _adapt_train_epoch(self, encoder_s, head_s, train_loader, optimizer, alpha):
        encoder_s.train()
        head_s.train()
        self.encoder.eval()
        self.head.eval()
        total_loss = 0
        total_num = 0
        total_logits = []
        for data, y_oracle in train_loader:
            data = data.to(self.device)
            student_logits = head_s(encoder_s(data))
            teacher_logits = self.head(self.encoder(data))
            loss, mask = self._pseudo_label_loss(student_logits, teacher_logits, alpha)
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

    def _adapt_eval_epoch(self, encoder_s, head_s, val_loader, alpha):
        encoder_s.eval()
        head_s.eval()
        self.encoder.eval()
        self.head.eval()
        total_loss = 0
        total_num = 0
        total_logits = []
        for data, y_oracle in val_loader:
            data = data.to(self.device)
            student_logits = head_s(encoder_s(data))
            teacher_logits = self.head(self.encoder(data))
            loss, mask = self._pseudo_label_loss(student_logits, teacher_logits, alpha)
            total_loss += loss.item() * mask.sum().item()
            total_num += mask.sum().item()
            total_logits.append(student_logits)
        total_loss /= total_num
        total_logits = torch.cat(total_logits)
        score = self.validator(target_train={"logits": total_logits})
        return total_loss, score

    @torch.no_grad()
    def _oracle_eval_epoch(self, encoder, head, val_loader):
        encoder.eval()
        head.eval()
        self.encoder.eval()
        self.head.eval()
        total_correct = 0
        total_pl_correct = 0
        total_num = 0
        for data, y in val_loader:
            data, y = data.to(self.device), y.to(self.device)
            output = head(encoder(data))

            pred = torch.argmax(output, dim=1)
            total_correct += torch.eq(pred, y).sum().item()

            pl = torch.argmax(self.head(self.encoder(data)), dim=1)
            total_pl_correct += torch.eq(pl, y).sum().item()
            total_num += data.shape[0]

        return total_correct / total_num, total_pl_correct / total_num

    def _adapt_train_eval(self, train_loader, confidence_q, args):
        alpha = self._calc_alpha(train_loader, confidence_q)

        encoder_s, head_s = deepcopy(self.encoder).to(self.device), deepcopy(self.head).to(self.device)

        optimizer = torch.optim.Adam(list(encoder_s.parameters()) + list(head_s.parameters()), lr=args.adapt_lr)
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_score = self._adapt_train_epoch(encoder_s, head_s, train_loader, optimizer, alpha)
            train_acc, pl_acc = self._oracle_eval_epoch(encoder_s, head_s, train_loader)

            print(f"Confidence q: {confidence_q} Epoch: {e} Train Loss: {train_loss} Train Acc: {train_acc} PL Acc: {pl_acc}") # Val Loss: {val_loss} Val Acc: {val_acc}")

            self.writer.add_scalar("Loss/train", train_loss, e)
            self.writer.add_scalar("Score/train", train_score, e)

        self.pl_acc_list.append(pl_acc)
        return encoder_s, head_s, 0


    @torch.no_grad()
    def _calc_alpha(self, loader, confidence_q):
        # find the quantile
        total_prob = []
        for data, _ in loader:
            data = data.to(self.device)
            logits = self.head(self.encoder(data))
            prob = torch.softmax(logits, dim=1)
            total_prob.append(prob)
        total_prob = torch.cat(total_prob)
        confidence = torch.amax(total_prob, 1) - torch.amin(total_prob, 1)
        alpha = torch.quantile(confidence, confidence_q)
        return alpha


    def _pseudo_label_loss(self, student_logits, teacher_logits, alpha):
        prob = torch.softmax(teacher_logits, dim=1)
        confidence = torch.amax(prob, 1) - torch.amin(prob, 1)
        mask = confidence >= alpha
        teacher_pred = torch.argmax(teacher_logits, dim=1)
        pseudo_loss = (F.nll_loss(F.log_softmax(student_logits, dim=1), teacher_pred, reduction='none') * mask).mean()
        return pseudo_loss, mask



    def adapt(self, domain_name, train_loader, confidence_q_list, args):
        # pseudo label train loader, val loader
        performance_dict = dict()
        for confidence_q in confidence_q_list:
            run_name = f"{args.method}_{confidence_q}_{args.random_seed}"
            self.writer = SummaryWriter(os.path.join(args.log_dir, args.dataset, domain_name, run_name))
            encoder_s, head_s, val_score = self._adapt_train_eval(train_loader, confidence_q, args)
            performance_dict[confidence_q] = {"encoder": encoder_s, "head": head_s, "score": val_score}

        best_val_score = -np.inf
        best_encoder, best_head = None, None
        for confidence_q, ckpt_dict in performance_dict.items():
            if ckpt_dict["score"] > best_val_score:
                best_encoder = ckpt_dict["encoder"]
                best_head = ckpt_dict["head"]
                best_val_score = ckpt_dict["score"]

        self._set_encoder_head(best_encoder, best_head)

    def _set_encoder_head(self, encoder, head):
        self.encoder = encoder.to(self.device)
        self.head = head.to(self.device)

    def get_encoder_head(self):
        return self.encoder, self.head

