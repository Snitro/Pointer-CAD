import dgl
import torch
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers.processing_utils import ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature


class PointerCADBatchFeature(BatchFeature):
    def to(self, *args, **kwargs) -> "PointerCADBatchFeature":
        for k, v in self.items():
            if isinstance(v, dgl.DGLGraph):
                self.data[k] = v.to(*args, **kwargs)
            if isinstance(v, list) and isinstance(v[0], torch.Tensor):
                self.data[k] = [t.to(*args, **kwargs) for t in v]
        return super().to(*args, **kwargs)


class PointerCADTokenizer(Qwen2Tokenizer):
    def __init__(self, *args, **kwargs):
        special_tokens_dict = {
            "additional_special_tokens": [
                "<|brep_edge_start|>",
                "<|brep_edge_end|>",
                "<|brep_edge_pad|>",
                "<|brep_face_start|>",
                "<|brep_face_end|>",
                "<|brep_face_pad|>",
                "<|cad_start|>",
                "<|cad_end|>",
                "<|cad_pad|>",
            ]
        }

        # Merge additional_special_tokens from kwargs.
        existing_specials = kwargs.pop("additional_special_tokens", [])
        special_tokens_dict["additional_special_tokens"].extend(
            t
            for t in existing_specials
            if t not in special_tokens_dict["additional_special_tokens"]
        )
        kwargs["additional_special_tokens"] = special_tokens_dict[
            "additional_special_tokens"
        ]

        super().__init__(*args, **kwargs)

        with open("./config/chat_template.jinja", "r", encoding="utf-8") as f:
            self.chat_template = f.read()

        self.brep_face_token = "<|brep_face_pad|>"
        self.brep_edge_token = "<|brep_edge_pad|>"
        self.cad_token = "<|cad_pad|>"

        self.add_special_tokens(special_tokens_dict)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path="Qwen/Qwen2.5-7B-Instruct", **kwargs
    ):
        # Call parent from_pretrained, then cast to subclass instance.
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        # Rebind __class__ to this subclass.
        tokenizer.__class__ = cls
        return tokenizer


class PointerCADProcessor(ProcessorMixin):

    attributes = ["tokenizer"]
    valid_kwargs = ["chat_template"]

    tokenizer_class = "Qwen2Tokenizer"

    def __init__(self, tokenizer: PointerCADTokenizer = None, **kwargs):
        super().__init__(tokenizer, chat_template=tokenizer.chat_template)
        self.brep_face_token = tokenizer.brep_face_token
        self.brep_edge_token = tokenizer.brep_edge_token
        self.cad_token = tokenizer.cad_token

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path="Qwen/Qwen2.5-7B-Instruct", **kwargs
    ):
        processor = cls(
            PointerCADTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        )
        return processor

    def __call__(
        self,
        text=None,
        breps=None,
        values=None,
        pointers=None,
        padding=True,
        max_length=None,
        return_tensors="pt",
    ):
        breps_inputs = {}
        cad_inputs = {}

        if not isinstance(text, list):
            text = [text]

        if breps is not None:
            breps_inputs = {"breps": breps}
            sub_breps = dgl.unbatch(breps)
            assert len(text) == len(sub_breps)

            for i in range(len(text)):
                ct: str = text[i]
                cg: dgl.DGLGraph = sub_breps[i]
                if cg is not None:
                    ct = ct.replace(
                        self.brep_edge_token,
                        self.brep_edge_token * int(cg.num_edges() / 2),
                        1,
                    )
                    text[i] = ct.replace(
                        self.brep_face_token, self.brep_face_token * cg.num_nodes(), 1
                    )
        else:
            for i in range(len(text)):
                ct: str = text[i]
                ct = ct.replace(self.brep_edge_token, "", 1)
                text[i] = ct.replace(self.brep_face_token, "", 1)

        if values is not None and pointers is not None:
            cad_inputs = {"values": values, "pointers": pointers}
            assert len(text) == len(values) == len(pointers)
            for i in range(len(text)):
                assert values[i].shape == pointers[i].shape
                text[i] = text[i].replace(
                    self.cad_token, self.cad_token * values[i].shape[0], 1
                )
        else:
            for i in range(len(text)):
                text[i] = text[i].replace(self.cad_token, "", 1)

        text_inputs = self.tokenizer(
            text,
            padding=padding,
            return_tensors=return_tensors,
            max_length=max_length,
            truncation=True if max_length is not None else None,
        )

        return PointerCADBatchFeature(data={**text_inputs, **breps_inputs, **cad_inputs})
