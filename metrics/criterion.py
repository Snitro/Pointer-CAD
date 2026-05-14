import torch
import torch.nn as nn
import torch.nn.functional as F

from misc import TOKEN, STANDARD_PLANES


class MAPELoss(nn.Module):
    """
    Mean Absolute Percentage Error (MAPE) Loss
    """

    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, pred, target):
        loss = torch.mean(torch.abs((pred - target) / (target + 1e-8)))
        return loss


class PointerCADCriterion(nn.Module):
    """
    Cross Entropy Loss for Pointer-CAD
    """

    def __init__(self, weight_value, weight_pointer, weight_scale):
        super(PointerCADCriterion, self).__init__()

        self.value_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.pointer_bce = nn.BCEWithLogitsLoss()
        self.scale_mape = MAPELoss()

        self.weight_value = weight_value
        self.weight_pointer = weight_pointer
        self.weight_scale = weight_scale

    def forward(
        self,
        pred_values,
        gt_values,
        pred_pointers,
        gt_pointers,
        pred_scales,
        gt_scales,
        gt_pointer_sources,
        pred_pointer_crv,
        pred_pointer_srf,
        pointer_tau,
        standard_plane_pointers,
    ):
        loss_value = 0
        for pred_value, gt_value in zip(pred_values, gt_values):
            ce_loss = self.value_ce(pred_value, gt_value[: pred_value.shape[0]])
            if torch.isnan(ce_loss):
                continue
            loss_value = loss_value + ce_loss
        loss_value = loss_value / len(gt_values)

        loss_pointer = 0
        gt_pointer_sources = gt_pointer_sources.copy()
        for (
            pred_pointer,
            gt_pointer,
            gt_pointer_source,
            gt_value,
            crv_pointer,
            srf_pointer,
        ) in zip(
            pred_pointers,
            gt_pointers,
            gt_pointer_sources,
            gt_values,
            pred_pointer_crv,
            pred_pointer_srf,
        ):
            if pred_pointer.shape[0] == 0:
                continue

            pointer_mask = gt_value[: pred_pointer.shape[0]] == TOKEN.index(
                "<|pointer_enable|>"
            )
            face_mask = torch.zeros_like(pointer_mask, dtype=torch.bool)
            face_mask[1:] = gt_value[: pred_pointer.shape[0] - 1] == TOKEN.index(
                "<|sketch_start|>"
            )
            face_mask &= pointer_mask
            curve_mask = (~face_mask) & pointer_mask

            face_idx = torch.nonzero(face_mask, as_tuple=False).squeeze(1)
            curve_idx = torch.nonzero(curve_mask, as_tuple=False).squeeze(1)

            face_num, curve_num = face_idx.numel(), curve_idx.numel()

            loss_pointer_srf = 0
            loss_pointer_crv = 0

            if face_num > 0:
                face_pred = pred_pointer[face_idx]
                face_target = torch.vstack(
                    [standard_plane_pointers, srf_pointer]
                ).type_as(face_pred)

                face_pred_norm = F.normalize(face_pred, p=2, dim=1, eps=1e-6)
                face_target_norm = F.normalize(face_target, p=2, dim=1, eps=1e-6)

                sim_face = (
                    torch.matmul(face_pred_norm, face_target_norm.T) * pointer_tau
                )
                gt_face = torch.zeros_like(sim_face, dtype=sim_face.dtype)

                for i, idx in enumerate(face_idx):
                    gt_face[
                        i, [x + len(STANDARD_PLANES) for x in gt_pointer_source[idx]]
                    ] = 1

                loss_pointer_srf = self.pointer_bce(sim_face, gt_face)

            if curve_num > 0:
                curve_pred = pred_pointer[curve_idx]
                curve_target = crv_pointer.type_as(curve_pred)

                curve_pred_norm = F.normalize(curve_pred, p=2, dim=1, eps=1e-6)
                curve_target_norm = F.normalize(curve_target, p=2, dim=1, eps=1e-6)

                sim_curve = (
                    torch.matmul(curve_pred_norm, curve_target_norm.T) * pointer_tau
                )
                gt_curve = torch.zeros_like(sim_curve, dtype=sim_curve.dtype)

                for i, idx in enumerate(curve_idx):
                    if len(gt_pointer_source[idx]) == 0:
                        gt_pointer_source[idx] = [
                            x
                            for x in gt_pointer_source[curve_idx[i - 1]]
                            if x != gt_pointer[curve_idx[i - 1]]
                        ]
                    gt_curve[i, gt_pointer_source[idx]] = 1

                loss_pointer_crv = self.pointer_bce(sim_curve, gt_curve)

            if face_num + curve_num > 0:
                loss_pointer = loss_pointer + (
                    loss_pointer_srf * face_num + loss_pointer_crv * curve_num
                ) / (face_num + curve_num)
        loss_pointer = loss_pointer / len(gt_pointer_sources)

        valid_scale_mask: torch.Tensor = pred_scales > 0
        if valid_scale_mask.sum() > 0:
            loss_scale: torch.Tensor = self.scale_mape(
                pred_scales[valid_scale_mask], gt_scales[valid_scale_mask]
            )
        else:
            loss_scale = 0

        loss = (
            loss_value * self.weight_value
            + loss_pointer * self.weight_pointer
            + loss_scale * self.weight_scale
        )
        if not torch.is_tensor(loss):
            loss = pointer_tau.sum() * 0.0

        return loss, {
            "value": (
                loss_value.item()
                if isinstance(loss_value, torch.Tensor)
                else loss_value
            ),
            "pointer": (
                loss_pointer.item()
                if isinstance(loss_pointer, torch.Tensor)
                else loss_pointer
            ),
            "scale": (
                loss_scale.item()
                if isinstance(loss_scale, torch.Tensor)
                else loss_scale
            ),
        }
