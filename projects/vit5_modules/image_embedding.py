import torch
from torch import nn
from vit5_modules.base import BaseEmbedding

class ImageEmbedding(BaseEmbedding):
    def __init__(self):
        super().init()
        self.layernorm_feat = nn.LayerNorm(self.hidden_size)
        self.linear_feat = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size
        )

        self.layernorm_box = nn.LayerNorm(self.hidden_size)
        self.linear_box = nn.Linear(
            in_features=4,
            out_features=self.hidden_size
        )

    def forward(self, batch):
        list_obj_boxes = batch["list_obj_boxes"]
        list_obj_feat = batch["list_obj_feat"]

        linear_box_out = self.layernorm_box(self.linear_box(list_obj_boxes))
        linear_feat_out = self.layernorm_feat(self.linear_feat(list_obj_feat))

        return linear_box_out + linear_feat_out