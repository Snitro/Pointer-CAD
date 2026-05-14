import torch
import copy
import torch.nn as nn
import os, sys

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))


from Cad_VLM.models.layers.attention import MultiHeadAttention
from Cad_VLM.models.layers.functional import FeedForwardLayer
from rich import print
from Cad_VLM.models.layers.utils_decode import generate_attention_mask
from Cad_VLM.models.utils import count_parameters
from typing import Optional


class TextAdaptiveLayer(nn.Module):
    """Adaptive layer for text embeddings."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int, dropout: float):
        super(TextAdaptiveLayer, self).__init__()

        self.sa_seq = MultiHeadAttention(
            input_dim=in_dim,
            embed_dim=in_dim,
            dropout=dropout,
            num_heads=num_heads,
        )

        self.norm_seq = nn.ModuleDict(
            {"norm_1": nn.LayerNorm(in_dim), "norm_2": nn.LayerNorm(in_dim)}
        )

        self.dp_seq = nn.ModuleDict(
            {"dropout_1": nn.Dropout(dropout), "dropout_2": nn.Dropout(dropout)}
        )

        self.ffl_seq = FeedForwardLayer(input_dim=in_dim)

        if in_dim != out_dim:
            self.downsampler = nn.Linear(in_dim, out_dim)

        self.attention_scores = dict()

    def forward(
        self,
        T: Optional[torch.Tensor],
        mask_prompt_dict: dict,
        metadata: bool = False,
    ):
        """
        Args:
            T (torch.Tensor): Text embedding tensor with shape (bs, num_seq, emb_dim).
            mask_prompt_dict (dict): Dictionary with keys "attn_mask" and "key_padding_mask".
            metadata (bool): Whether to return attention metadata.
        """
        T2 = self.norm_seq["norm_1"](T)
        T2, T_score = self.sa_seq(
            T2,
            T2,
            T2,
            key_padding_mask=mask_prompt_dict["key_padding_mask"],
            attn_mask=mask_prompt_dict["attn_mask"],
        )

        T = T + self.dp_seq["dropout_1"](T2)
        T2 = self.norm_seq["norm_2"](T)

        T = T + self.dp_seq["dropout_2"](self.ffl_seq(T2))

        if hasattr(self, "downsampler"):
            T = self.downsampler(T)

        if metadata:
            self.attention_scores["text_sattn"] = T_score
            return T, self.attention_scores
        else:
            return T, None

    def total_parameters(self, description=False, in_millions=False):
        num_params = count_parameters(self, description)
        if in_millions:
            num_params_million = num_params / 1_000_000  # Convert to millions
            print(f"Number of Parameters: {num_params_million:.1f}M")
        else:
            num_params = count_parameters(self, description)
            print(f"Number of Parameters: {num_params}")

    @staticmethod
    def from_config(config):
        return TextAdaptiveLayer(**config)
