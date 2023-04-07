import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pytorch_adapt.validators import SNDValidator


class UncertaintyAggregatedTeacher:
    def __init__(self, model, device="cpu"):
        self.device = device
        self.student = model.to(self.device)
        self.teacher = deepcopy(model).to(self.device)
        self.validator = SNDValidator()

    def _adapt_train_epoch(self, model, train_loader, optimizer, alpha):
        model.train()
        total_loss = 0
        total_num = 0
        total_logits = []
        for data, _ in train_loader:
            data = data.to(self.device)
            student_logits = model(data)
            teacher_logits = self.teacher(data)
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

    @torch.no_grad()
    def _adapt_eval_epoch(self, model, val_loader, alpha):
        model.eval()
        total_loss = 0
        total_num = 0
        total_logits = []
        for data, _ in val_loader:
            data = data.to(self.device)
            student_logits = model(data)
            teacher_logits = self.teacher(data)
            loss, mask = self._pseudo_label_loss(student_logits, teacher_logits, alpha)
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

        # TODO: update teacher using current teacher and current student
        self._update_teacher(train_loader)
        # TODO: alpha = self._calc_alpha(train_loader, confidence_q) # calculate alpha with the updated teacher
        alpha = self._calc_alpha(train_loader, confidence_q)
        model = deepcopy(self.student) # initialize model to optimize with current student (optimized in previous domain)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.adapt_lr)
        best_val_loss = np.inf
        best_val_score = None
        best_model = None
        patience = 5
        staleness = 0
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_score = self._adapt_train_epoch(model, train_loader, optimizer, alpha)
            train_acc = self._oracle_eval_epoch(model, train_loader)
            if not val_loader:
                best_model = deepcopy(model)
                best_val_score = train_score
                print(f"Confidence q: {confidence_q} Epoch: {e} Train Loss: {train_loss} Train Acc: {train_acc}")
                continue
            val_loss, val_score = self._adapt_eval_epoch(model, val_loader, alpha)
            val_acc = self._oracle_eval_epoch(model, val_loader)

            print(f"Confidence q: {confidence_q} Epoch: {e} Train Loss: {train_loss} Val Loss: {val_loss} Val Acc: {val_acc}")

            self.writer.add_scalar("Loss/train", train_loss, e)
            self.writer.add_scalar("Loss/val", val_loss, e)
            self.writer.add_scalar("Score/train", train_score, e)
            self.writer.add_scalar("Score/val", val_score, e)

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
            logits = self.teacher(data)
            prob = torch.softmax(logits, dim=1)
            total_prob.append(prob)
        total_prob = torch.cat(total_prob)
        confidence = torch.amax(total_prob, 1) - torch.amin(total_prob, 1)
        alpha = torch.quantile(confidence, confidence_q)
        print(f"alpha: {alpha} average max prob: {torch.mean(torch.amax(total_prob, 1))}")
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


    def _negative_entropy(self, logits):
        entropies = -torch.sum(
            torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1), dim=1
        )
        return -torch.mean(entropies)


    def _update_teacher(self, loader):
        self.teacher.eval()
        self.student.eval()

        teacher_logits = []
        student_logits = []
        for data, _ in loader:
            data = data.to(self.device)
            teacher_logits.append(self.teacher(data).detach().cpu())
            student_logits.append(self.student(data).detach().cpu())
        teacher_logits = torch.cat(teacher_logits)
        student_logits = torch.cat(student_logits)
        teacher_conf = self._negative_entropy(teacher_logits)
        student_conf = self._negative_entropy(student_logits)
        print("Teacher confidence:", teacher_conf)
        print("Student confidence:", student_conf)
        beta = torch.exp(-torch.log(torch.tensor(2.)) * teacher_conf / student_conf) # favor high momentum
        print("Beta:", beta)
        for teacher_param, param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.data = beta * teacher_param + (1 - beta) * param

        for m2, m1 in zip(self.teacher.named_modules(), self.student.named_modules()):
            if ('bn' in m2[0]) and ('bn' in m1[0]):
                bn2, bn1 = m2[1].state_dict(), m1[1].state_dict()
                bn2['running_mean'].data.copy_(bn1['running_mean'].data)
                bn2['running_var'].data.copy_(bn1['running_var'].data)
                bn2['num_batches_tracked'].data.copy_(bn1['num_batches_tracked'].data)


    def adapt(self, domain_name, train_loader, confidence_q_list, args, val_loader=None):
        # pseudo label train loader, val loader
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

        self.student = deepcopy(best_model)

    def get_model(self):
        return self.student

