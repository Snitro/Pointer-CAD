import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers.modeling_rope_utils import dynamic_rope_update
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2RotaryEmbedding
from transformers.modeling_outputs import BaseModelOutputWithPast

from misc import TOKEN, STANDARD_PLANES, MAX_VECTOR_LENGTH
from .brep_embed import UVNetEmbedder


class PointerCAD(nn.Module):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}

    def __init__(
        self,
        qwen_model="Qwen/Qwen2.5-7B-Instruct",
        crv_channels=12,
        surf_channels=8,
        pointer_size=128,
        max_tau=100.0,
        bit=8,
        dtype=torch.bfloat16,
    ):
        super().__init__()

        self.pointer_size = pointer_size
        self.dtype = dtype

        self.model: Qwen2Model = Qwen2Model.from_pretrained(
            qwen_model, torch_dtype=dtype, attn_implementation="flash_attention_2"
        )
        self.brep = UVNetEmbedder(
            crv_channels,
            surf_channels,
            pointer_size,
            pointer_size,
            self.model.config.hidden_size,
        )
        self.value_embedding = nn.Embedding(
            len(TOKEN) + 2**bit,
            self.model.config.hidden_size,
            TOKEN.index("<|padding|>"),
        )
        self.pointer_projection = nn.Linear(
            self.pointer_size, self.model.config.hidden_size, bias=False
        )
        self.value_head = nn.Linear(
            self.model.config.hidden_size, len(TOKEN) + 2**bit, bias=False, dtype=dtype
        )
        self.pointer_head = nn.Linear(
            self.model.config.hidden_size, self.pointer_size, bias=False, dtype=dtype
        )
        self.scale_head = nn.Linear(self.model.config.hidden_size, 1, dtype=dtype)
        self.scale_activation = nn.Softplus()
        self.cad_position_embedding = nn.Embedding(
            2, self.model.config.hidden_size, dtype=dtype
        )

        self.pointer_tau = nn.Parameter(
            torch.log(torch.tensor(1 / 0.07, dtype=torch.float32))
        )
        self.pointer_tau_max = math.log(max_tau)
        self.standard_plane_pointer = nn.Parameter(
            torch.randn(len(STANDARD_PLANES), pointer_size)
        )

        with torch.no_grad():
            self.cad_position_embedding.weight.zero_()

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.model = get_peft_model(self.model, lora_config)

    @torch.no_grad()
    def clamp_parameters(self):
        with torch.no_grad():
            self.pointer_tau.clamp_(max=self.pointer_tau_max)

    def forward(
        self,
        input_ids,
        attention_mask,
        breps,
        values,
        pointers,
        logits_to_keep=0,
        **kwargs,
    ):
        text_embeds: torch.Tensor = self.model.get_input_embeddings()(input_ids)

        crv_mask: torch.Tensor = input_ids == 151667  # TODO: Move token IDs to config.
        srf_mask: torch.Tensor = input_ids == 151670  # TODO: Move token IDs to config.
        cad_mask: torch.Tensor = input_ids == 151673  # TODO: Move token IDs to config.
        scale_idx = cad_mask.to(torch.long).argmax(dim=1)

        if breps.num_edges() > 0:
            pointer_crv, pointer_srf, feat_crv, feat_srf = self.brep(breps)

            feat_crv_clipped = [
                (
                    feats[: mask.sum().item()]
                    if feats.shape[0] > mask.sum().item()
                    else feats
                )
                for feats, mask in zip(feat_crv, crv_mask)
            ]
            feat_srf_clipped = [
                (
                    feats[: mask.sum().item()]
                    if feats.shape[0] > mask.sum().item()
                    else feats
                )
                for feats, mask in zip(feat_srf, srf_mask)
            ]

            crv_mask_expanded = crv_mask.unsqueeze(-1).expand_as(text_embeds)
            srf_mask_expanded = srf_mask.unsqueeze(-1).expand_as(text_embeds)

            feat_crv = torch.vstack(feat_crv_clipped).type_as(text_embeds)
            feat_srf = torch.vstack(feat_srf_clipped).type_as(text_embeds)

            assert crv_mask.sum() == feat_crv.shape[0]
            assert srf_mask.sum() == feat_srf.shape[0]

            text_embeds = text_embeds.masked_scatter(crv_mask_expanded, feat_crv)
            text_embeds = text_embeds.masked_scatter(srf_mask_expanded, feat_srf)
        else:
            pointer_crv = [
                torch.zeros(
                    (0, self.pointer_size), dtype=torch.float, device=text_embeds.device
                )
                for _ in range(len(pointers))
            ]
            pointer_srf = pointer_crv

        assert len(values) == len(pointers)  # same batch size
        if len(values) > 0:
            cad_feature = []
            for idx, (
                batch_pointer_crv,
                batch_pointer_srf,
                batch_value,
                batch_pointer,
                batch_cad_mask,
            ) in enumerate(zip(pointer_crv, pointer_srf, values, pointers, cad_mask)):
                total_cad_num = batch_cad_mask.sum()
                if total_cad_num == 0:
                    continue
                valid_srf_crv_mask = batch_pointer[:total_cad_num] >= 0
                srf_mask = torch.zeros(
                    (total_cad_num), dtype=torch.bool, device=valid_srf_crv_mask.device
                )
                srf_mask[1:] = batch_value[: total_cad_num - 1] == TOKEN.index(
                    "<|sketch_start|>"
                )
                srf_mask &= valid_srf_crv_mask
                crv_mask: torch.Tensor = ~srf_mask & valid_srf_crv_mask
                plane_mask: torch.Tensor = (
                    batch_pointer[:total_cad_num] >= -len(STANDARD_PLANES)
                ) & ~valid_srf_crv_mask
                valid_mask: torch.Tensor = plane_mask | valid_srf_crv_mask

                valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
                srf_idx = torch.nonzero(srf_mask, as_tuple=False).squeeze(1)
                crv_idx = torch.nonzero(crv_mask, as_tuple=False).squeeze(1)
                plane_idx = torch.nonzero(plane_mask, as_tuple=False).squeeze(1)

                pointer_valid = torch.empty(
                    (valid_idx.size(0), self.pointer_size), device=batch_pointer.device
                )

                if srf_idx.numel() > 0:
                    valid_srf_idx = torch.nonzero(
                        srf_mask[valid_idx], as_tuple=False
                    ).squeeze(1)
                    pointer_valid[valid_srf_idx] = batch_pointer_srf[
                        batch_pointer[srf_idx]
                    ]

                if crv_idx.numel() > 0:
                    valid_crv_idx = torch.nonzero(
                        crv_mask[valid_idx], as_tuple=False
                    ).squeeze(1)
                    pointer_valid[valid_crv_idx] = batch_pointer_crv[
                        batch_pointer[crv_idx]
                    ]

                if plane_idx.numel() > 0:
                    valid_plane_idx = torch.nonzero(
                        plane_mask[valid_idx], as_tuple=False
                    ).squeeze(1)
                    pointer_valid[valid_plane_idx] = self.standard_plane_pointer[
                        batch_pointer[plane_idx] + len(STANDARD_PLANES)
                    ]

                value_embeds: torch.Tensor = self.value_embedding(
                    batch_value[:total_cad_num]
                )

                if TOKEN.index("<|model_end|>") in batch_value[:total_cad_num]:
                    scale_idx[idx] += (
                        (batch_value[:total_cad_num] == TOKEN.index("<|model_end|>"))
                        .to(torch.long)
                        .argmax()
                    )
                elif TOKEN.index("<|part_end|>") in batch_value[:total_cad_num]:
                    scale_idx[idx] += (
                        (batch_value[:total_cad_num] == TOKEN.index("<|part_end|>"))
                        .to(torch.long)
                        .argmax()
                    )
                else:
                    scale_idx[idx] = -1

                if valid_idx.numel():
                    pointer_embeds = self.pointer_projection(pointer_valid)
                    value_embeds_idx = torch.nonzero(valid_mask).squeeze(1)

                    assert value_embeds_idx.shape[0] == pointer_embeds.shape[0]

                    value_embeds = value_embeds.index_add(
                        0, value_embeds_idx, pointer_embeds
                    )

                cad_feature.append(value_embeds)

            if cad_feature:
                cad_mask_expanded = cad_mask.unsqueeze(-1).expand_as(text_embeds)
                cad_feature = torch.vstack(cad_feature).type_as(text_embeds)
                assert cad_mask.sum() == cad_feature.shape[0]
                text_embeds = text_embeds.masked_scatter(cad_mask_expanded, cad_feature)
            else:
                assert not torch.any(cad_mask)

        last_feature_class = cad_mask[:, -1].clone()
        for i in range(last_feature_class.shape[0]):
            if last_feature_class[i]:
                last_value = values[i][cad_mask[i].sum() - 1]
                if last_value == TOKEN.index(
                    "<|model_end|>"
                ) or last_value == TOKEN.index("<|part_end|>"):
                    last_feature_class[i] = False
        last_feature_class |= input_ids[:, -1] == 151671
        shifted_cad_mask = torch.cat(
            [cad_mask[:, 1:], last_feature_class.unsqueeze(1)], dim=1
        )
        position_embeds = self.cad_position_embedding(
            shifted_cad_mask.type_as(input_ids)
        )

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            inputs_embeds=text_embeds + position_embeds,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        hidden_states = hidden_states[:, slice_indices, :]

        shifted_cad_mask = shifted_cad_mask[:, slice_indices]
        cad_states = hidden_states[shifted_cad_mask]
        num_cad_per_batch = shifted_cad_mask.sum(dim=1).tolist()

        pred_values = self.value_head(cad_states)
        pred_pointers = self.pointer_head(cad_states)

        shifted_scale_idx = scale_idx - 1
        scale_mask = shifted_scale_idx > 0
        scale_states = hidden_states[scale_mask].gather(
            1,
            shifted_scale_idx[scale_mask]
            .view(-1, 1, 1)
            .expand(-1, 1, hidden_states.size(2)),
        )
        pred_scales_valid = self.scale_activation(
            self.scale_head(scale_states)
        ).squeeze()
        pred_scales = torch.full_like(scale_mask, -1, dtype=hidden_states.dtype)
        pred_scales[scale_mask] = pred_scales_valid

        return (
            torch.split(pred_values, num_cad_per_batch),
            torch.split(pred_pointers, num_cad_per_batch),
            pred_scales,
            pointer_crv,
            pointer_srf,
            self.pointer_tau.exp().clone(),
            self.standard_plane_pointer.clone(),
        )

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        breps: dgl.DGLGraph,
        temperature_value=1.0,
        temperature_pointer=1.0,
        max_steps: int = MAX_VECTOR_LENGTH,
        mode="argmax",  # can be "argmax" or "sample"
        **kwargs,
    ):
        assert mode in [
            "argmax",
            "sample",
        ], f"Invalid mode: {mode}. Must be 'argmax' or 'sample'."

        self.eval()

        batch_size = input_ids.shape[0]
        generated_txt = input_ids.clone()
        generated_mask = attention_mask.clone()
        generated_value = [
            torch.zeros((0), dtype=input_ids.dtype, device=input_ids.device)
            for _ in range(batch_size)
        ]
        generated_pointer = [
            torch.zeros((0), dtype=input_ids.dtype, device=input_ids.device)
            for _ in range(batch_size)
        ]
        generated_scale = (
            torch.ones((batch_size), dtype=self.dtype, device=input_ids.device) * -1
        )
        past_key_values = None

        text_embeds: torch.Tensor = self.model.get_input_embeddings()(generated_txt)

        crv_mask = generated_txt == 151667  # TODO: Move token IDs to config.
        srf_mask = generated_txt == 151670  # TODO: Move token IDs to config.

        if breps.num_edges() > 0:
            pointer_crv, pointer_srf, feat_crv, feat_srf = self.brep(breps)

            feat_crv_clipped = [
                (
                    feats[: mask.sum().item()]
                    if feats.shape[0] > mask.sum().item()
                    else feats
                )
                for feats, mask in zip(feat_crv, crv_mask)
            ]
            feat_srf_clipped = [
                (
                    feats[: mask.sum().item()]
                    if feats.shape[0] > mask.sum().item()
                    else feats
                )
                for feats, mask in zip(feat_srf, srf_mask)
            ]

            crv_mask_expanded = crv_mask.unsqueeze(-1).expand_as(text_embeds)
            srf_mask_expanded = srf_mask.unsqueeze(-1).expand_as(text_embeds)

            feat_crv = torch.vstack(feat_crv_clipped).type_as(text_embeds)
            feat_srf = torch.vstack(feat_srf_clipped).type_as(text_embeds)

            assert crv_mask.sum() == feat_crv.shape[0]
            assert srf_mask.sum() == feat_srf.shape[0]

            text_embeds = text_embeds.masked_scatter(crv_mask_expanded, feat_crv)
            text_embeds = text_embeds.masked_scatter(srf_mask_expanded, feat_srf)
        else:
            pointer_crv = [
                torch.zeros(
                    (0, self.pointer_size), dtype=torch.float, device=text_embeds.device
                )
                for _ in range(batch_size)
            ]
            pointer_srf = pointer_crv

        pointer_srf = [
            torch.vstack([self.standard_plane_pointer, srf_pointer])
            for srf_pointer in pointer_srf
        ]

        inputs_embeds = text_embeds + self.cad_position_embedding(
            (input_ids == 151671).type_as(input_ids)
        )

        for step in range(max_steps):
            if past_key_values is None:
                outputs: BaseModelOutputWithPast = self.model(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    attention_mask=generated_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                outputs: BaseModelOutputWithPast = self.model(
                    input_ids=None,
                    inputs_embeds=inputs_embeds[:, -1:, :],
                    attention_mask=generated_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state[
                :, -1, :
            ]  # shape: batch * hidden_size

            finished_batch_num = 0
            generated_txt = torch.cat(
                [
                    generated_txt,
                    151672
                    * torch.ones(
                        (batch_size, 1),
                        device=generated_txt.device,
                        dtype=generated_txt.dtype,
                    ),
                ],
                dim=-1,
            )
            for idx in range(batch_size):
                if generated_txt[idx][-2] == 151671 or (
                    (generated_txt[idx][-2] == 151673)
                    and (generated_value[idx][-1] != TOKEN.index("<|model_end|>"))
                    and (generated_value[idx][-1] != TOKEN.index("<|part_end|>"))
                ):
                    if mode == "sample":
                        pred_value = torch.multinomial(
                            F.softmax(
                                self.value_head(hidden_states[idx]) / temperature_value,
                                dim=0,
                            ),
                            num_samples=1,
                        ).squeeze(0)
                    else:
                        pred_value = torch.argmax(
                            self.value_head(hidden_states[idx]), dim=0
                        )

                    if step > 0 and pred_value == TOKEN.index("<|pointer_enable|>"):
                        pred_pointer: torch.Tensor = self.pointer_head(
                            hidden_states[idx]
                        )
                        if generated_value[idx][-1] == TOKEN.index("<|sketch_start|>"):
                            pred = pred_pointer.unsqueeze(0).expand(
                                pointer_srf[idx].size(0), -1
                            )
                            cos_sim = (
                                F.cosine_similarity(pred, pointer_srf[idx], dim=1)
                                * self.pointer_tau.exp()
                            )
                            index = (
                                torch.argmax(cos_sim)
                                if mode == "argmax"
                                else torch.multinomial(
                                    F.softmax(cos_sim / temperature_pointer, dim=0),
                                    num_samples=1,
                                ).squeeze(0)
                            )
                            pred_pointer = index - len(STANDARD_PLANES)
                        else:
                            if pointer_crv[idx].shape[0] > 0:
                                pred = pred_pointer.unsqueeze(0).expand(
                                    pointer_crv[idx].size(0), -1
                                )
                                cos_sim = (
                                    F.cosine_similarity(pred, pointer_crv[idx], dim=1)
                                    * self.pointer_tau.exp()
                                )
                                pred_pointer = (
                                    torch.argmax(cos_sim)
                                    if mode == "argmax"
                                    else torch.multinomial(
                                        F.softmax(cos_sim / temperature_pointer, dim=0),
                                        num_samples=1,
                                    ).squeeze(0)
                                )
                            else:
                                pred_pointer = torch.tensor(
                                    -len(STANDARD_PLANES) - 1,
                                    dtype=generated_pointer[idx].dtype,
                                    device=generated_pointer[idx].device,
                                )
                                pred_value = torch.tensor(
                                    TOKEN.index("<|pointer_disable|>"),
                                    dtype=pred_value.dtype,
                                    device=pred_value.device,
                                )
                    else:
                        pred_pointer = torch.tensor(
                            -len(STANDARD_PLANES) - 1,
                            dtype=generated_pointer[idx].dtype,
                            device=generated_pointer[idx].device,
                        )

                    generated_value[idx] = torch.cat(
                        [generated_value[idx], pred_value.unsqueeze(0)]
                    )
                    generated_pointer[idx] = torch.cat(
                        [generated_pointer[idx], pred_pointer.unsqueeze(0)]
                    )
                    generated_txt[idx][-1] = 151673

                    if pred_value.unsqueeze(0).item() in [
                        TOKEN.index("<|model_end|>"),
                        TOKEN.index("<|part_end|>"),
                    ]:
                        generated_scale[idx] = self.scale_activation(
                            self.scale_head(hidden_states[idx])
                        )
                else:
                    finished_batch_num += 1

            if finished_batch_num == batch_size:
                break

            inputs_embeds_next: torch.Tensor = self.model.get_input_embeddings()(
                generated_txt[:, -1]
            )  # shape: batch_size * hidden_size
            for idx in range(batch_size):
                if generated_txt[idx][-1] == 151673:
                    value_embeds = self.value_embedding(
                        generated_value[idx][-1]
                    )  # shape: hiddensize
                    generated_pointer_idx = generated_pointer[idx][-1]

                    if generated_pointer_idx >= -len(STANDARD_PLANES):
                        pointer_embeds = (
                            pointer_srf[idx][
                                generated_pointer_idx + len(STANDARD_PLANES)
                            ]
                            if generated_value[idx][-2]
                            == TOKEN.index("<|sketch_start|>")
                            else pointer_crv[idx][generated_pointer_idx]
                        )
                        pointer_embeds = self.pointer_projection(pointer_embeds)
                        value_embeds = value_embeds + pointer_embeds

                    inputs_embeds_next[idx] = value_embeds
            inputs_embeds_next = inputs_embeds_next + self.cad_position_embedding(
                (generated_txt[:, -1] == 151673).type_as(input_ids)
            )
            inputs_embeds = torch.cat(
                [inputs_embeds, inputs_embeds_next.unsqueeze(1)], dim=1
            )

            generated_mask_pad = torch.ones(
                (batch_size, generated_txt.shape[1] - generated_mask.shape[1]),
                device=generated_mask.device,
                dtype=generated_mask.dtype,
            )
            generated_mask = torch.cat([generated_mask, generated_mask_pad], dim=-1)

        return generated_value, generated_pointer, generated_scale

    def get_param_groups(self, base_lr: float, tau_lr: float, weight_decay: float):
        other_params, tau_params = [], []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith("pointer_tau"):
                tau_params.append(p)
            else:
                other_params.append(p)

        groups = []
        if other_params:
            groups.append(
                {"params": other_params, "lr": base_lr, "weight_decay": weight_decay}
            )
        if tau_params:
            groups.append({"params": tau_params, "lr": tau_lr, "weight_decay": 0.0})
        return groups
