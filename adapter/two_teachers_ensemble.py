import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pytorch_adapt.validators import SNDValidator


class TwoTeachersEnsemble:
    def __init__(self, model, device="cpu"):
        self.device = device
        self.model = model.to(self.device)
        self.src_teacher = deepcopy(model).to(self.device)
        self.validator = SNDValidator()

    def _adapt_train_epoch(self, model, train_loader, optimizer, alpha):
        model.train()
        total_loss = 0
        total_num = 0
        total_logits = []
        for data, _ in train_loader:
            data = data.to(self.device)
            student_logits = model(data)
            cur_teacher_logits = self.model(data)
            src_teacher_logits = self.src_teacher(data)
            teacher_ensemble = (cur_teacher_logits + src_teacher_logits) / 2
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
    def _adapt_eval_epoch(self, model, val_loader, alpha):
        model.eval()
        total_loss = 0
        total_num = 0
        total_logits = []
        for data, _ in val_loader:
            data = data.to(self.device)
            student_logits = model(data)
            cur_teacher_logits = self.model(data)
            src_teacher_logits = self.src_teacher(data)
            teacher_ensemble = (cur_teacher_logits + src_teacher_logits) / 2
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

    def _adapt_train_eval(self, train_loader, confidence_q, args, val_loader=None):
        alpha = self._calc_alpha(train_loader, confidence_q)
        model = deepcopy(self.model).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.adapt_lr)
        best_val_loss = np.inf
        best_val_score = None
        best_model = None, None
        patience = 5
        staleness = 0
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_score = self._adapt_train_epoch(model, train_loader, optimizer, alpha)
            train_acc = self._oracle_eval_epoch(model, train_loader)
            if not val_loader:
                best_model = deepcopy(model)
                best_val_score = train_score
                print(
                    f"Confidence q: {confidence_q} Epoch: {e} Train Loss: {round(train_loss, 3)} Train Acc: {round(train_acc, 3)}")
                continue

            val_loss, val_score = self._adapt_eval_epoch(model, val_loader, alpha)
            val_acc = self._oracle_eval_epoch(model, val_loader)

            print(f"Confidence q: {confidence_q} Epoch: {e} Train Loss: {round(train_loss, 3)} Train Acc: {round(train_acc, 3)} Val Loss: {round(val_loss, 3)} Val Acc: {round(val_acc, 3)}")

            self.writer.add_scalar("Loss/train", train_loss, e)
            self.writer.add_scalar("Loss/val", val_loss, e)
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
        # cur_teacher_pred = []
        # src_teacher_pred = []
        for data, _ in loader:
            data = data.to(self.device)
            cur_teacher_logits = self.model(data)
            src_teacher_logits = self.src_teacher(data)
            print("cur teacher prob:", F.softmax(cur_teacher_logits, dim=1)[0])
            print("src teacher prob:", F.softmax(src_teacher_logits, dim=1)[0])
            # cur_teacher_pred.append(torch.argmax(cur_teacher_logits, dim=1))
            # src_teacher_pred.append(torch.argmax(src_teacher_logits, dim=1))
            teacher_ensemble = (cur_teacher_logits + src_teacher_logits) / 2
            prob = torch.softmax(teacher_ensemble, dim=1)
            total_prob.append(prob)
        total_prob = torch.cat(total_prob)
        confidence = torch.amax(total_prob, 1) - torch.amin(total_prob, 1)
        alpha = torch.quantile(confidence, confidence_q)
        # cur_teacher_pred = torch.cat(cur_teacher_pred)
        # src_teacher_pred = torch.cat(src_teacher_pred)
        # print("Agreement rate:", torch.eq(cur_teacher_pred, src_teacher_pred).sum().item() / len(cur_teacher_pred))
        return alpha

    def _pseudo_label_loss(self, y, y_target, alpha):
        prob = torch.softmax(y_target, dim=1)
        confidence = torch.amax(prob, 1) - torch.amin(prob, 1)
        mask = confidence >= alpha
        pseudo_labels = torch.argmax(prob, dim=1)
        pseudo_loss = (F.nll_loss(F.log_softmax(y, dim=1), pseudo_labels, reduction='none') * mask).mean()
        return pseudo_loss, mask, pseudo_labels



    def adapt(self, domain_name, train_loader, confidence_q_list, args, val_loader=None):
        performance_dict = dict()
        for confidence_q in confidence_q_list:
            run_name = f"{args.method}_{confidence_q}_{args.random_seed}"
            self.writer = SummaryWriter(os.path.join(args.log_dir, args.dataset, domain_name, run_name))
            model, val_score = self._adapt_train_eval(train_loader, confidence_q, args, val_loader)
            performance_dict[confidence_q] = {"model": model, "score": val_score}

        best_val_score = -np.inf
        best_model = None
        for confidence_q, ckpt_dict in performance_dict.items():
            if ckpt_dict["score"] > best_val_score:
                best_model = ckpt_dict["model"]
                best_val_score = ckpt_dict["score"]

        self.model = deepcopy(best_model).to(self.device)

    def get_model(self):
        return self.model

