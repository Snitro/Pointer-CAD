import torch
from collections import deque
import torch.nn.functional as F

from misc import TOKEN, STANDARD_PLANES


class SmoothedMetric:
    def __init__(self, window_size=20, ignore_invalid=True):
        self.window_size = window_size
        self.ignore_invalid = ignore_invalid
        self.values = deque(maxlen=window_size)

    def update(self, value):
        if value < 0:
            if self.ignore_invalid:
                return
            value = 0.0
        self.values.append(value)

    def average(self):
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def max(self):
        if not self.values:
            return 0.0
        return max(self.values)

    def reset(self):
        self.values.clear()


class ValueAccuracyCalculator:
    def __init__(
        self, bit=8, padding_token=TOKEN.index("<|padding|>"), value_token=len(TOKEN)
    ):
        self.padding_token = padding_token
        self.value_token = value_token

        self.tolerance = round(
            3 * (2 ** (bit - 8))
        )  # 8-bit tolerance is 3, 10-bit tolerance is 12, 12-bit tolerance is 48

    def calculateFromProbability2D(self, predProb, targetLabel):
        # Get the predicted classes
        return self.calculateFromLabel2D(
            [pred.cpu().argmax(dim=-1) for pred in predProb],
            [f.cpu() for f in targetLabel],
        )

    def calculateFromLabel2D(self, predLabel, targetLabel):
        """
        predLabel: list of tensor
        target: list of tensor
        """
        accuracy_t, accuracy_k, accuracy_v = 0, 0, 0
        num_t, num_k, num_v = 0, 0, 0
        for pred, target in zip(predLabel, targetLabel):
            if len(target.shape) == 1:
                acc_t, acc_k, acc_v = self.calculateFromLabel(
                    predLabel=pred, targetLabel=target
                )
            else:
                acc_t, acc_k, acc_v = -1, -1, -1
                for i in range(target.shape[0]):
                    acc_t_, acc_k_, acc_v_ = self.calculateFromLabel(
                        predLabel=pred, targetLabel=target[i]
                    )
                    acc_t = max(acc_t, acc_t_)
                    acc_k = max(acc_k, acc_k_)
                    acc_v = max(acc_v, acc_v_)

            if acc_t >= 0:
                accuracy_t += acc_t
                num_t += 1
            if acc_k >= 0:
                accuracy_k += acc_k
                num_k += 1
            if acc_v >= 0:
                accuracy_v += acc_v
                num_v += 1

        return (
            accuracy_t / num_t if num_t > 0 else -1,
            accuracy_k / num_k if num_k > 0 else -1,
            accuracy_v / num_v if num_v > 0 else -1,
        )

    def calculateFromLabel(self, predLabel, targetLabel):
        """
        pred: tensor of shape (N)
        target: tensor of shape (N)
        """

        N_pred = predLabel.shape[0]
        N_gt = targetLabel.shape[0]

        if N_pred > N_gt:
            predLabel = predLabel[:N_gt]
        elif N_pred < N_gt:
            inf_tensor = torch.full((N_gt - N_pred,), float("inf"))
            predLabel = torch.cat([predLabel, inf_tensor])

        mask_v = (targetLabel >= self.value_token) & (targetLabel != self.padding_token)
        mask_k = (targetLabel < self.value_token) & (targetLabel != self.padding_token)
        mask_v = mask_v.to(targetLabel.device)
        mask_k = mask_k.to(targetLabel.device)

        # Calculate the number of correct predictions (removing the padding)
        correct_v: torch.Tensor = (
            (torch.abs(predLabel - targetLabel) < self.tolerance) * 1 * mask_v
        )
        correct_k: torch.Tensor = (predLabel == targetLabel) * 1 * mask_k

        correct_v = correct_v.sum()
        correct_k = correct_k.sum()
        correct = correct_v + correct_k

        # Calculate the total number of predictions excluding the discard token
        total_v = mask_v.sum()
        total_k = mask_k.sum()
        total = total_v + total_k

        # Calculate the accuracy
        ret_total, ret_k, ret_v = -1, -1, -1

        if total > 0:
            ret_total = float(correct) * 100 / float(total)
        if total_v > 0:
            ret_v = float(correct_v) * 100 / float(total_v)
        if total_k > 0:
            ret_k = float(correct_k) * 100 / float(total_k)

        return ret_total, ret_k, ret_v


class PointerAccuracyCalculator:
    def __init__(self):
        pass

    def calculateFromLabel2D(self, pred_pointers, gt_pointers):
        # Get the predicted classes
        accuracy = 0
        num = 0
        for pred_pointer, gt_pointer in zip(pred_pointers, gt_pointers):
            acc = self.calculateFromLabel(pred_pointer.cpu(), gt_pointer)

            if acc >= 0:
                accuracy += acc
                num += 1

        return accuracy / num if num > 0 else -1

    def calculateFromPointer2D(
        self,
        pred_pointers,
        gt_pointers,
        gt_values,
        pred_pointers_crv,
        pred_pointers_srf,
        pointer_tau,
        standard_plane_pointers,
    ):
        # Get the predicted classes
        accuracy = 0
        num = 0
        for (
            pred_pointer,
            gt_pointer,
            gt_value,
            pred_pointer_crv,
            pred_pointer_srf,
        ) in zip(
            pred_pointers, gt_pointers, gt_values, pred_pointers_crv, pred_pointers_srf
        ):
            acc = self.calculateFromPointer(
                pred_pointer.cpu(),
                gt_pointer,
                gt_value.cpu(),
                pred_pointer_crv.cpu(),
                pred_pointer_srf.cpu(),
                pointer_tau.cpu(),
                standard_plane_pointers.cpu(),
            )

            if acc >= 0:
                accuracy += acc
                num += 1

        return accuracy / num if num > 0 else -1

    def calculateFromPointer(
        self,
        pred_pointers,
        gt_pointers,
        gt_values,
        pred_pointers_crv,
        pred_pointers_srf,
        pointer_tau,
        standard_plane_pointers,
    ):
        """
        predLabel: list of tensor
        target: list of tensor
        """
        pred_labels = []
        for i in range(pred_pointers.shape[0]):
            pointer_ids = gt_pointers[i]
            if isinstance(pointer_ids, list):
                if gt_values[i - 1] == TOKEN.index("<|sketch_start|>"):
                    candidate_pointers = torch.vstack(
                        [standard_plane_pointers, pred_pointers_srf]
                    )
                    pred = (
                        pred_pointers[i]
                        .unsqueeze(0)
                        .expand(candidate_pointers.size(0), -1)
                    )
                    cos_sim = (
                        F.cosine_similarity(pred, candidate_pointers, dim=1)
                        * pointer_tau
                    )
                    index = torch.argmax(cos_sim)
                    pred_labels.append(index.item() - len(STANDARD_PLANES))
                else:
                    pred = (
                        pred_pointers[i]
                        .unsqueeze(0)
                        .expand(pred_pointers_crv.size(0), -1)
                    )
                    cos_sim = (
                        F.cosine_similarity(pred, pred_pointers_crv, dim=1)
                        * pointer_tau
                    )
                    index = torch.argmax(cos_sim)
                    pred_labels.append(index.item())

            else:
                pred_labels.append(pointer_ids)

        return self.calculateFromLabel(pred_labels, gt_pointers)

    def calculateFromLabel(self, predLabel, targetLabel):
        N_pred = len(predLabel)
        N_gt = len(targetLabel)
        N_min = min(N_pred, N_gt)
        predLabel = predLabel[:N_min]
        targetLabel = targetLabel[:N_min]

        correct, total = 0, 0.0
        for idx in range(N_min):
            pred, target = predLabel[idx], targetLabel[idx]
            if not (isinstance(target, list) or isinstance(target, set)):
                continue
            total += 1

            if len(target) == 0:
                target = [x for x in targetLabel[idx - 1] if x != predLabel[idx - 1]]
                targetLabel[idx] = target

            correct += 1 if pred in target else 0

        # Calculate the accuracy
        if total > 0:
            return correct * 100 / total
        else:
            return -1


class ScaleAccuracyCalculator:
    def calculateFromBatch(self, pred, gt):
        mask: torch.Tensor = pred > 0
        if mask.sum() == 0:
            return -1

        error = torch.clamp(
            torch.abs((pred[mask] - gt[mask]) / (gt[mask] + 1e-8)), 0, 1
        )
        return 100 * (1 - torch.mean(error))
