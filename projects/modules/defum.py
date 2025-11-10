import torch
import numpy as np
from torch import nn

from utils.registry import registry
from icecream import ic

from projects.modules.attention import DeFumAttention

class DeFUM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = registry.get_config("model_attributes")
        self.defum_config = self.config["defum"]
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")

        self.hidden_size = self.config["hidden_size"]
        self.common_dim = self.config["feature_dim"]

        #--layer
        self.defum_attention = DeFumAttention(input_size=self.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.common_dim,
            nhead=self.defum_config["nhead"],
            activation=self.defum_config["activation"],
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.defum_config["num_layers"])
        self.LayerNorm = nn.LayerNorm(normalized_shape=self.hidden_size)
        

    #-- FUNCTION
    def cal_relative_score(self, dv_i, dv_j):
        return max(dv_i, dv_j) / min(dv_i, dv_j)


    def cal_relative_depth_map(self, depth_visual_entity):
        """
            :params depth_visual_entity:  BS, num_obj + num_ocr, 1
        """
        relative_matrix = [
            [self.cal_relative_score(dv_i, dv_j) for dv_i in depth_visual_entity ]
            for dv_j in depth_visual_entity
        ]
        return torch.tensor(relative_matrix)


    
    #-- FORWARD
    def forward(self, batch):
        #~ Load features
        list_ocr_feat = batch["list_ocr_feat"]
        list_obj_feat = batch["list_obj_feat"]
        list_ocr_depth_feat = batch["list_ocr_depth_feat"]
        list_obj_depth_feat = batch["list_obj_depth_feat"]

        #~ Attention Mask
        ocr_mask = batch["ocr_mask"]
        obj_mask = batch["obj_mask"]
        attention_mask = torch.concat(
            [ocr_mask, obj_mask],
            dim=1
        )

        #~ Visual Entities / Depth Visual Entities (Concat obj_feat and ocr_feat) (obj is the bridge link its adjacent scene texts)
        visual_entity = torch.concat(
            [list_ocr_feat, list_obj_feat],
            dim=1
        )
        depth_visual_entity = torch.concat(
            [list_ocr_depth_feat, list_obj_depth_feat],
            dim=1
        )
        relative_depth_map = []
        for dv_item in depth_visual_entity:
            relative_depth_map_item = self.cal_relative_depth_map(dv_item)
            relative_depth_map.append(torch.log(relative_depth_map_item))
        R = torch.stack(relative_depth_map).to(torch.float32).to(self.device)

        #~ Depth Aware Self Attention
        attention_scores = self.defum_attention(
            visual_entity=visual_entity, 
            relative_depth_map=relative_depth_map, 
            attention_mask=attention_mask
        )

        #~ Transformer Encoder
        inputs = self.LayerNorm(visual_entity + attention_scores)
        depth_aware_visual_feat = self.transformer_encoder(
            inputs, 
            src_key_padding_mask=attention_mask.bool()
        )
        return depth_aware_visual_feat


        

