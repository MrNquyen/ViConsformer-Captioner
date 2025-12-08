import torch
from torch import nn
from vit5_modules.base import BaseEmbedding
from transformers.models.t5.modeling_t5 import T5LayerNorm

class SceneTextEmbedding(BaseEmbedding):
    def __init__(self, ):
        super().init()
        self.layernorm_feat = nn.T5LayerNorm(self.hidden_size)
        self.layernorm_box = nn.T5LayerNorm(self.hidden_size)

        self.linear_feat = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size
        )
        self.linear_box = nn.Linear(
            in_features=4,
            out_features=self.hidden_size
        )
        self.linear_ocr_vit5 = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size
        )


    def forward(
        self, 
        batch,
        ocr_tokens_embed_vit5
    ):
        list_ocr_boxes = batch["list_ocr_boxes"]
        list_ocr_feat = batch["list_ocr_feat"]

        linear_box_out = self.layernorm_box(self.linear_box(list_ocr_boxes))
        linear_feat_out = self.layernorm_feat(self.linear_feat(list_ocr_feat))
        linear_ocr_vit5_out = self.linear_ocr_vit5(ocr_tokens_embed_vit5)

        return linear_box_out + linear_feat_out + linear_ocr_vit5_out

