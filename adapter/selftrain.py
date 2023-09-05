import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_adapt.validators import IMValidator # SNDValidator


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
        # total_pl_correct = 0
        # total_num_wo_thres = 0
        total_logits = []
        for data, y_oracle in train_loader:
            data = data.to(self.device)
            y_oracle = y_oracle.to(self.device)
            student_logits = head_s(encoder_s(data))
            teacher_logits = self.head(self.encoder(data))
            loss, mask = self._pseudo_label_loss(student_logits, teacher_logits, alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mask.sum().item()
            total_num += mask.sum().item()
            pl = torch.argmax(teacher_logits, dim=1)
            # total_pl_correct += torch.eq(pl, y_oracle).sum().item()
            # total_num_wo_thres += data.shape[0]
            total_logits.append(student_logits)
        total_loss /= total_num
        # total_pl_correct /= total_num_wo_thres
        total_logits = torch.cat(total_logits)
        score = self.validator(target_train={"logits": total_logits})
        return total_loss, score # , total_pl_correct

    def _adapt_eval_epoch(self, encoder_s, head_s, val_loader, alpha):
        encoder_s.eval()
        head_s.eval()
        self.encoder.eval()
        self.head.eval()
        total_loss = 0
        total_num = 0
        # total_pl_correct = 0
        # total_num_wo_thres = 0
        total_logits = []
        for data, y_oracle in val_loader:
            data = data.to(self.device)
            y_oracle = y_oracle.to(self.device)
            student_logits = head_s(encoder_s(data))
            teacher_logits = self.head(self.encoder(data))
            loss, mask = self._pseudo_label_loss(student_logits, teacher_logits, alpha)
            total_loss += loss.item() * mask.sum().item()
            total_num += mask.sum().item()
            pl = torch.argmax(teacher_logits, dim=1)
            # total_pl_correct += torch.eq(pl, y_oracle).sum().item()
            # total_num_wo_thres += data.shape[0]
            total_logits.append(student_logits)
        total_loss /= total_num
        # total_pl_correct /= total_num_wo_thres
        total_logits = torch.cat(total_logits)
        score = self.validator(target_train={"logits": total_logits})
        return total_loss, score # , total_pl_correct

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

    def _measure_pl_acc(self, loader):
        self.encoder.eval()
        self.head.eval()
        total_correct = 0
        total_num = 0
        for data, y in loader:
            data, y = data.to(self.device), y.to(self.device)
            output = self.head(self.encoder(data))
            pred = torch.argmax(output, dim=1)
            total_correct += torch.eq(pred, y).sum().item()
            total_num += data.shape[0]
        return total_correct / total_num

    def _adapt_train_eval(self, train_loader, confidence_q, args):
        # pseudo_train_loader = self._pseudo_label(train_loader, self.encoder, self.head, confidence_q)
        # pseudo_val_loader = self._pseudo_label(val_loader, self.encoder, self.head, confidence_q)
        # TODO: measure pseudo-label quality
        # pl_acc = self._measure_pl_acc(train_loader)
        # self.pl_acc_list.append(pl_acc)
        alpha = self._calc_alpha(train_loader, confidence_q)

        encoder_s, head_s = deepcopy(self.encoder).to(self.device), deepcopy(self.head).to(self.device)

        optimizer = torch.optim.Adam(list(encoder_s.parameters()) + list(head_s.parameters()), lr=args.adapt_lr)
        # best_val_loss = np.inf
        # best_val_score = None
        # best_encoder, best_head = None, None
        # patience = 20
        # staleness = 0
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_score = self._adapt_train_epoch(encoder_s, head_s, train_loader, optimizer, alpha)
            # val_loss, val_score = self._adapt_eval_epoch(encoder_s, head_s, val_loader, alpha)
            # val_acc = self._oracle_eval_epoch(encoder_s, head_s, val_loader)
            train_acc, pl_acc = self._oracle_eval_epoch(encoder_s, head_s, train_loader)

            print(f"Confidence q: {confidence_q} Epoch: {e} Train Loss: {train_loss} Train Acc: {train_acc} PL Acc: {pl_acc}") # Val Loss: {val_loss} Val Acc: {val_acc}")

            self.writer.add_scalar("Loss/train", train_loss, e)
            # self.writer.add_scalar("Loss/val", val_loss, e)
            self.writer.add_scalar("Score/train", train_score, e)
            # self.writer.add_scalar("Score/val", val_score, e)


            # if val_loss < best_val_loss:
            #     best_encoder = deepcopy(encoder_s)
            #     best_head = deepcopy(head_s)
            #     best_val_loss = val_loss
            #     best_val_score = val_score
            #     staleness = 0
            # else:
            #     staleness += 1
            #
            # if staleness > patience:
            #     break

        # return best_encoder, best_head, best_val_score
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
        #
        # print(f"average max prob: {torch.mean(torch.amax(total_prob, dim=1))}")
        # print(f"average entropy: {torch.mean(torch.sum(total_prob * torch.log(total_prob), dim=1))}")
        return alpha


    def _pseudo_label_loss(self, student_logits, teacher_logits, alpha):
        # student_logits = head_s(encoder_s(data))
        # teacher_logits = self.head(self.encoder(data))
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

