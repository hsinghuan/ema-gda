import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_adapt.validators import SNDValidator


class SelfTrainer:
    def __init__(self, encoder, head, device="cpu"):
        self.device = device
        self._set_encoder_head(encoder, head)
        self.validator = SNDValidator()

    def _adapt_train_epoch(self, encoder_s, head_s, train_loader, optimizer, alpha):
        encoder_s.train()
        head_s.train()
        total_loss = 0
        total_num = 0
        total_logits = []
        for data, _ in train_loader:
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
        score = self.validator(target_train={"preds": total_logits})
        return total_loss, score

    def _adapt_eval_epoch(self, encoder_s, head_s, val_loader, alpha):
        encoder_s.eval()
        head_s.eval()
        total_loss = 0
        total_num = 0
        total_logits = []
        for data, _ in val_loader:
            data = data.to(self.device)
            student_logits = head_s(encoder_s(data))
            teacher_logits = self.head(self.encoder(data))
            loss, mask = self._pseudo_label_loss(student_logits, teacher_logits, alpha)
            total_loss += loss.item() * mask.sum().item()
            total_num += mask.sum().item()
            total_logits.append(student_logits)
        total_loss /= total_num
        total_logits = torch.cat(total_logits)
        score = self.validator(target_train={"preds": total_logits})
        return total_loss, score

    @torch.no_grad()
    def _oracle_eval_epoch(self, encoder, head, val_loader):
        encoder.eval()
        head.eval()
        total_correct = 0
        total_num = 0
        for data, y in val_loader:
            data, y = data.to(self.device), y.to(self.device)
            output = head(encoder(data))

            pred = torch.argmax(output, dim=1)
            total_correct += torch.eq(pred, y).sum().item()
            total_num += data.shape[0]

        return total_correct / total_num

    def _adapt_train_eval(self, train_loader, val_loader, confidence_q, args):
        # pseudo_train_loader = self._pseudo_label(train_loader, self.encoder, self.head, confidence_q)
        # pseudo_val_loader = self._pseudo_label(val_loader, self.encoder, self.head, confidence_q)
        alpha = self._calc_alpha(train_loader, confidence_q)

        encoder_s, head_s = deepcopy(self.encoder).to(self.device), deepcopy(self.head).to(self.device)

        optimizer = torch.optim.Adam(list(encoder_s.parameters()) + list(head_s.parameters()), lr=args.adapt_lr)
        best_val_loss = np.inf
        best_val_score = None
        best_encoder, best_head = None, None
        patience = 5
        staleness = 0
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_score = self._adapt_train_epoch(encoder_s, head_s, train_loader, optimizer, alpha)
            val_loss, val_score = self._adapt_eval_epoch(encoder_s, head_s, val_loader, alpha)
            val_acc = self._oracle_eval_epoch(encoder_s, head_s, val_loader)

            print(f"Confidence q: {confidence_q} Epoch: {e} Train Loss: {train_loss} Val Loss: {val_loss} Val Acc: {val_acc}")

            self.writer.add_scalar("Loss/train", train_loss, e)
            self.writer.add_scalar("Loss/val", val_loss, e)
            self.writer.add_scalar("Score/train", train_score, e)
            self.writer.add_scalar("Score/val", val_score, e)

            if val_loss < best_val_loss:
                best_encoder = deepcopy(encoder_s)
                best_head = deepcopy(head_s)
                best_val_loss = val_loss
                best_val_score = val_score
                staleness = 0
            else:
                staleness += 1

            if staleness > patience:
                break

        return best_encoder, best_head, best_val_score

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
        print(f"alpha: {alpha} average max prob: {torch.mean(torch.amax(total_prob, 1))}")
        return alpha
        # # make pseudo-labeled dataset
        # data_list = []
        # lbl_list = []
        # for data, _ in loader:
        #     data = data.to(self.device)
        #     logits = head_t(encoder_t(data))
        #     prob = torch.softmax(logits, dim=1)
        #     confidence = torch.amax(prob, 1) - torch.amin(prob, 1)
        #     indices = confidence >= alpha
        #     pred = torch.argmax(prob, dim=1)
        #     data_list.append(data[indices])
        #     lbl_list.append(pred[indices])
        # data_tensor = torch.cat(data_list, dim=0)
        # print("pseudo labeled dataset size:", data_tensor.shape)
        # lbl_tensor = torch.cat(lbl_list, dim=0)
        # pseudo_loader = DataLoader(TensorDataset(data_tensor, lbl_tensor), batch_size=128, shuffle=True)
        #
        # return pseudo_loader

    def _pseudo_label_loss(self, student_logits, teacher_logits, alpha):
        # student_logits = head_s(encoder_s(data))
        # teacher_logits = self.head(self.encoder(data))
        prob = torch.softmax(teacher_logits, dim=1)
        confidence = torch.amax(prob, 1) - torch.amin(prob, 1)
        mask = confidence >= alpha
        teacher_pred = torch.argmax(teacher_logits, dim=1)
        pseudo_loss = (F.nll_loss(F.log_softmax(student_logits, dim=1), teacher_pred, reduction='none') * mask).mean()
        return pseudo_loss, mask



    def adapt(self, domain_name, train_loader, val_loader, confidence_q_list, args):
        # pseudo label train loader, val loader
        performance_dict = dict()
        for confidence_q in confidence_q_list:
            run_name = f"{args.method}_{confidence_q}_{args.random_seed}"
            self.writer = SummaryWriter(os.path.join(args.log_dir, args.dataset, domain_name, run_name))
            encoder_s, head_s, val_score = self._adapt_train_eval(train_loader, val_loader, confidence_q, args)
            performance_dict[confidence_q] = {"encoder": encoder_s, "head": head_s, "score": val_score}

        best_val_score = -np.inf
        best_encoder, best_head = None, None
        for confidence_q, ckpt_dict in performance_dict.items():
            if ckpt_dict["score"] > best_val_score:
                best_encoder = ckpt_dict["encoder"]
                best_head = ckpt_dict["head"]

        self._set_encoder_head(best_encoder, best_head)

    def _set_encoder_head(self, encoder, head):
        self.encoder = encoder.to(self.device)
        self.head = head.to(self.device)

    def get_encoder_head(self):
        return self.encoder, self.head

