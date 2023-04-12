import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_adapt.validators import SNDValidator


class HierarchicalTeacherSigmoid:
    def __init__(self, model, device="cpu"):
        self.device = device
        self.model = model.to(self.device)
        self.inter_teacher = deepcopy(model).to(self.device)
        self.validator = SNDValidator()

    def _adapt_train_epoch(self, model, inter_teacher, intra_teacher, train_loader, optimizer, alpha, lamb):
        model.train()
        total_loss = 0
        total_pseudo_loss = 0
        total_consistency_loss = 0
        total_logits = []
        for data, _ in train_loader:
            data = data.to(self.device)
            model_logits = model(data)
            inter_teacher_logits = inter_teacher(data)
            intra_teacher_logits = intra_teacher(data)
            pseudo_loss, _ = self._pseudo_label_loss(model_logits, inter_teacher_logits, alpha)
            consistency_loss = self._consistency_loss(model_logits, intra_teacher_logits)
            loss = pseudo_loss + lamb * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_pseudo_loss += pseudo_loss.item()
            total_consistency_loss += consistency_loss.item()
            total_logits.append(model_logits)

            self._update_intra_teacher(intra_teacher, model)

        total_loss /= len(train_loader)
        total_pseudo_loss /= len(train_loader)
        total_consistency_loss /= len(train_loader)
        total_logits = torch.cat(total_logits)
        score = self.validator(target_train={"preds": total_logits})
        return total_loss, total_pseudo_loss, total_consistency_loss, score

    @torch.no_grad()
    def _adapt_eval_epoch(self, model, inter_teacher, intra_teacher, val_loader, alpha, lamb):
        model.eval()
        total_loss = 0
        total_pseudo_loss = 0
        total_consistency_loss = 0
        total_logits = []
        for data, _ in val_loader:
            data = data.to(self.device)
            model_logits = model(data)
            inter_teacher_logits = inter_teacher(data)
            intra_teacher_logits = intra_teacher(data)
            pseudo_loss, _ = self._pseudo_label_loss(model_logits, inter_teacher_logits, alpha)
            consistency_loss = self._consistency_loss(model_logits, intra_teacher_logits)
            loss = pseudo_loss + lamb * consistency_loss
            total_loss += loss.item()
            total_pseudo_loss += pseudo_loss.item()
            total_consistency_loss += consistency_loss.item()
            total_logits.append(model_logits)
        total_loss /= len(val_loader)
        total_pseudo_loss /= len(val_loader)
        total_consistency_loss /= len(val_loader)
        total_logits = torch.cat(total_logits)
        score = self.validator(target_train={"preds": total_logits})
        return total_loss, total_pseudo_loss, total_consistency_loss, score

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

    def _adapt_train_eval(self, train_loader, confidence_q, sharpness, lamb, args, val_loader=None):
        model = deepcopy(self.model)
        inter_teacher = deepcopy(self.inter_teacher)
        self._update_inter_teacher(inter_teacher, model, train_loader, sharpness)
        alpha = self._calc_alpha(train_loader, confidence_q)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.adapt_lr)
        best_val_loss = np.inf
        best_val_score = None
        best_model = None
        patience = args.adapt_epochs # 5
        staleness = 0
        intra_teacher = deepcopy(model)
        for e in range(1, args.adapt_epochs + 1):
            train_loss, train_pseudo_loss, train_con_loss, train_score = self._adapt_train_epoch(model, inter_teacher, intra_teacher, train_loader, optimizer, alpha, lamb)
            train_acc = self._oracle_eval_epoch(model, train_loader)
            if not val_loader:
                best_model = deepcopy(model)
                best_val_score = train_score
                print(f"Confidence q: {confidence_q} Epoch: {e} Train Loss: {round(train_loss, 3)} Train Pseudo Loss: {round(train_pseudo_loss, 3)} Train Con Loss: {round(train_con_loss, 3)} Train Acc: {round(train_acc, 3)}")
                self._update_intra_teacher(intra_teacher, model)
                continue
            val_loss, val_pseudo_loss, val_con_loss, val_score = self._adapt_eval_epoch(model, inter_teacher, intra_teacher, val_loader, alpha, lamb)
            val_acc = self._oracle_eval_epoch(model, val_loader)

            print(f"Confidence q: {confidence_q} Epoch: {e} Train Loss: {round(val_loss, 3)} Train Pseudo Loss: {round(train_pseudo_loss, 3)} Train Con Loss: {round(train_con_loss, 3)} Train Acc: {round(train_acc, 3)} \n\
                                                            Val Loss: {round(val_loss, 3)} Val Pseudo Loss: {round(val_pseudo_loss, 3)} Val Con Loss: {round(val_con_loss, 3)} Val Acc: {round(val_acc, 3)}")
            self.writer.add_scalar("Loss/train", train_loss, e)
            self.writer.add_scalar("Loss/val", val_loss, e)
            self.writer.add_scalar("Pseudo Loss/train", train_pseudo_loss, e)
            self.writer.add_scalar("Pseudo Loss/val", val_pseudo_loss, e)
            self.writer.add_scalar("Con Loss/train", train_con_loss, e)
            self.writer.add_scalar("Con Loss/val", val_con_loss, e)
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



        return best_model, best_val_score, inter_teacher

    @torch.no_grad()
    def _calc_alpha(self, loader, confidence_q):
        # find the quantile
        total_prob = []
        for data, _ in loader:
            data = data.to(self.device)
            logits = self.inter_teacher(data)
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

    def _consistency_loss(self, student_logits, teacher_logits):
        return torch.mean(torch.norm(F.softmax(student_logits, dim=1) - F.softmax(teacher_logits, dim=1), 2, dim=1))

    def _entropy(self, logits):
        entropies = -torch.sum(
            torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1), dim=1
        )
        return torch.mean(entropies)


    def _update_inter_teacher(self, inter_teacher, model, loader, sharpness):
        inter_teacher.eval()
        model.eval()

        teacher_logits = []
        model_logits = []
        for data, _ in loader:
            data = data.to(self.device)
            teacher_logits.append(inter_teacher(data).detach().cpu())
            model_logits.append(model(data).detach().cpu())
        teacher_logits = torch.cat(teacher_logits)
        model_logits = torch.cat(model_logits)
        teacher_ent = self._entropy(teacher_logits)
        model_ent = self._entropy(model_logits)
        # momentum = torch.clip(0.5 + slope * (model_ent - teacher_ent), 0, 1)
        momentum = torch.sigmoid((model_ent - teacher_ent) / sharpness)
        print(f"Inter Teacher Entropy: {teacher_ent} Current Model Entropy: {model_ent}: Momentum: {momentum} Entropy Diff: {model_ent - teacher_ent} Sharpness: {sharpness}")
        for teacher_param, param in zip(inter_teacher.parameters(), model.parameters()):
            teacher_param.data = momentum * teacher_param + (1 - momentum) * param

        for m2, m1 in zip(inter_teacher.named_modules(), model.named_modules()):
            if ('bn' in m2[0]) and ('bn' in m1[0]):
                bn2, bn1 = m2[1].state_dict(), m1[1].state_dict()
                bn2['running_mean'].data.copy_(bn1['running_mean'].data)
                bn2['running_var'].data.copy_(bn1['running_var'].data)
                bn2['num_batches_tracked'].data.copy_(bn1['num_batches_tracked'].data)

    def _update_intra_teacher(self, intra_teacher, model):
        for teacher_param, param in zip(intra_teacher.parameters(), model.parameters()):
            teacher_param.data = 0.95 * teacher_param + (1 - 0.95) * param

        for m2, m1 in zip(intra_teacher.named_modules(), model.named_modules()):
            if ('bn' in m2[0]) and ('bn' in m1[0]):
                bn2, bn1 = m2[1].state_dict(), m1[1].state_dict()
                bn2['running_mean'].data.copy_(bn1['running_mean'].data)
                bn2['running_var'].data.copy_(bn1['running_var'].data)
                bn2['num_batches_tracked'].data.copy_(bn1['num_batches_tracked'].data)

    def adapt(self, domain_name, train_loader, confidence_q_list, sharpness_list, lambda_list, args, val_loader=None):
        # pseudo label train loader, val loader
        performance_dict = dict()
        for confidence_q in confidence_q_list:
            for sharpness in sharpness_list:
                for lamb in lambda_list:
                    run_name = f"{args.method}_{confidence_q}_{sharpness}_{lamb}_{args.random_seed}"
                    self.writer = SummaryWriter(os.path.join(args.log_dir, args.dataset, domain_name, run_name))
                    model, val_score, inter_teacher = self._adapt_train_eval(train_loader, confidence_q, sharpness, lamb, args, val_loader)
                    performance_dict[(confidence_q, sharpness, lamb)] = {"model": model, "score": val_score, "inter_teacher": inter_teacher}

        best_val_score = -np.inf
        best_confidence_q, best_sharpness, best_lamb = None, None, None
        best_model, best_inter_teacher = None, None
        for (confidence_q, sharpness, lamb), ckpt_dict in performance_dict.items():
            print(f"Confidence Q: {confidence_q} Sharpness: {sharpness} Lambda: {lamb} Score:", ckpt_dict["score"])
            if ckpt_dict["score"] > best_val_score:
                best_model = ckpt_dict["model"]
                best_inter_teacher = ckpt_dict["inter_teacher"]
                best_confidence_q = confidence_q
                best_sharpness = sharpness
                best_lamb = lamb
                best_val_score = ckpt_dict["score"]

        print(f"Best Confidence Q: {best_confidence_q} Best Sharpness: {best_sharpness} Best Lamb: {best_lamb}")
        # use all data to train
        if val_loader:
            # TODO: merge two dataloaders
            data_all, y_all = [], []
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
            run_name = f"{args.method}_all_{best_confidence_q}_{best_sharpness}_{best_lamb}_{args.random_seed}"
            self.writer = SummaryWriter(os.path.join(args.log_dir, args.dataset, domain_name, run_name))
            best_model, _, best_inter_teacher = self._adapt_train_eval(loader, best_confidence_q, best_sharpness, best_lamb, args)

        self.model = deepcopy(best_model)
        self.inter_teacher = deepcopy(best_inter_teacher)

    def get_model(self):
        return self.model

