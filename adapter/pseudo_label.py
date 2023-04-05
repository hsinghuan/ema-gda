import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pytorch_adapt.validators import SNDValidator


class PseudoLabelTrainer:
    def __init__(self, model, src_train_loader, src_val_loader, device="cpu"):
        self.device = device
        self.model = model.to(self.device)
        self.src_train_loader = src_train_loader
        self.src_val_loader = src_val_loader
        self.validator = SNDValidator()

    def _adapt_train_epoch(self, model, train_loader, optimizer, alpha, tradeoff):
        model.train()
        total_loss = 0
        total_src_loss = 0
        total_tgt_loss = 0
        total_logits = []
        len_dataloader = min(len(train_loader), len(self.src_train_loader))
        src_iter = iter(self.src_train_loader)
        tgt_iter = iter(train_loader)

        for _ in range(len_dataloader):
            src_data, src_y = next(src_iter)
            src_data, src_y = src_data.to(self.device), src_y.to(self.device)
            src_logits = model(src_data)
            src_loss = F.nll_loss(F.log_softmax(src_logits, dim=1), src_y)

            tgt_data, _ = next(tgt_iter)
            tgt_data = tgt_data.to(self.device)
            tgt_logits = model(tgt_data)
            tgt_loss, mask, _ = self._pseudo_label_loss(tgt_logits, tgt_logits, alpha)

            loss = src_loss + tradeoff * tgt_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_src_loss += src_loss.item()
            total_tgt_loss += tgt_loss.item()
            total_logits.append(tgt_logits)

        total_loss /= len_dataloader
        total_src_loss /= len_dataloader
        total_tgt_loss /= len_dataloader
        total_logits = torch.cat(total_logits)
        score = self.validator(target_train={"preds": total_logits})
        return total_loss, total_src_loss, total_tgt_loss, score

    @torch.no_grad()
    def _adapt_eval_epoch(self, model, val_loader, alpha, tradeoff):
        model.eval()
        total_loss = 0
        total_src_loss = 0
        total_tgt_loss = 0
        total_logits = []
        len_dataloader = min(len(val_loader), len(self.src_val_loader))
        src_iter = iter(self.src_val_loader)
        tgt_iter = iter(val_loader)

        for _ in range(len_dataloader):
            src_data, src_y = next(src_iter)
            src_data, src_y = src_data.to(self.device), src_y.to(self.device)
            src_logits = model(src_data)
            src_loss = F.nll_loss(F.log_softmax(src_logits, dim=1), src_y)

            tgt_data, _ = next(tgt_iter)
            tgt_data = tgt_data.to(self.device)
            tgt_logits = model(tgt_data)
            tgt_loss, mask, _ = self._pseudo_label_loss(tgt_logits, tgt_logits, alpha)

            loss = src_loss + tradeoff * tgt_loss

            total_loss += loss.item()
            total_src_loss += src_loss.item()
            total_tgt_loss += tgt_loss.item()
            total_logits.append(tgt_logits)

        total_loss /= len_dataloader
        total_src_loss /= len_dataloader
        total_tgt_loss /= len_dataloader
        total_logits = torch.cat(total_logits)
        score = self.validator(target_train={"preds": total_logits})
        return total_loss, total_src_loss, total_tgt_loss, score

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

    def _adapt_train_eval(self, train_loader, val_loader, confidence_q, tradeoff, args):
        alpha = self._calc_alpha(train_loader, confidence_q)
        model = deepcopy(self.model).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.adapt_lr)
        best_val_loss = np.inf
        best_val_score = None
        best_model = None, None
        patience = 5
        staleness = 0
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_src_loss, train_tgt_loss, train_score = self._adapt_train_epoch(model, train_loader, optimizer, alpha, tradeoff)
            val_loss, val_src_loss, val_tgt_loss, val_score = self._adapt_eval_epoch(model, val_loader, alpha, tradeoff)
            train_acc = self._oracle_eval_epoch(model, train_loader)
            val_acc = self._oracle_eval_epoch(model, val_loader)

            print(f"Confidence q: {confidence_q} Tradeoff: {tradeoff} Epoch: {e} Train Loss: {round(train_loss, 3)} Train Src Loss: {round(train_src_loss, 3)} Train Tgt Loss: {round(train_tgt_loss, 3)} Train Acc: {round(train_acc, 3)} Val Loss: {round(val_loss, 3)} Val Src Loss: {round(val_src_loss, 3)} Val Tgt Loss: {round(val_tgt_loss, 3)} Val Acc: {round(val_acc, 3)}")

            self.writer.add_scalar("Loss/train", train_loss, e)
            self.writer.add_scalar("Loss/val", val_loss, e)
            self.writer.add_scalar("Src Loss/train", train_src_loss, e)
            self.writer.add_scalar("Src Loss/val", val_src_loss, e)
            self.writer.add_scalar("Tgt Loss/train", train_tgt_loss, e)
            self.writer.add_scalar("Tgt Loss/val", val_tgt_loss, e)
            self.writer.add_scalar("Score/train", train_score, e)
            self.writer.add_scalar("Score/val", val_score, e)
            self.writer.add_scalar("Acc/train", train_acc, e)
            self.writer.add_scalar("Acc/val", val_acc, e)



            if val_loss < best_val_loss:
                best_model = deepcopy(model)
                best_val_loss = val_loss
                best_val_score = val_score
                staleness = 0
            else:
                staleness += 1

            if staleness > patience:
                break

        return best_model, best_val_score


    @torch.no_grad()
    def _calc_alpha(self, loader, confidence_q):
        # find the quantile
        total_prob = []
        for data, _ in loader:
            data = data.to(self.device)
            logits = self.model(data)
            prob = torch.softmax(logits, dim=1)
            total_prob.append(prob)
        total_prob = torch.cat(total_prob)
        confidence = torch.amax(total_prob, 1) - torch.amin(total_prob, 1)
        alpha = torch.quantile(confidence, confidence_q)
        print(f"alpha: {alpha} average max prob: {torch.mean(torch.amax(total_prob, 1))}")
        return alpha

    def _pseudo_label_loss(self, y, y_target, alpha):
        prob = torch.softmax(y_target, dim=1)
        confidence = torch.amax(prob, 1) - torch.amin(prob, 1)
        mask = confidence >= alpha
        pseudo_labels = torch.argmax(prob, dim=1)
        pseudo_loss = (F.nll_loss(F.log_softmax(y, dim=1), pseudo_labels, reduction='none') * mask).mean()
        return pseudo_loss, mask, pseudo_labels



    def adapt(self, domain_name, train_loader, val_loader, confidence_q_list, tradeoff_list, args):
        performance_dict = dict()
        for confidence_q in confidence_q_list:
            for tradeoff in tradeoff_list:
                run_name = f"{args.method}_{confidence_q}_{tradeoff}_{args.random_seed}"
                self.writer = SummaryWriter(os.path.join(args.log_dir, args.dataset, domain_name, run_name))
                model, val_score = self._adapt_train_eval(train_loader, val_loader, confidence_q, tradeoff, args)
                performance_dict[confidence_q] = {"model": model, "score": val_score}

        best_val_score = -np.inf
        best_model = None
        for confidence_q, ckpt_dict in performance_dict.items():
            if ckpt_dict["score"] > best_val_score:
                best_model = ckpt_dict["model"]

        self.model = deepcopy(best_model).to(self.device)

    def get_model(self):
        return self.model

