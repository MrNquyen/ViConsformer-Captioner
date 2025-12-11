import torch
import copy
from torch import nn
from typing import List
from torch.nn import functional as F

from tqdm import tqdm
from time import time
from icecream import ic

from projects.vit5_modules_new.attention import (
    ViConstituentModule, 
    ScaledDotProductAttention, 
)
from utils.registry import registry


#============== Utils ================
def clones(module, N):
    """
        Produce N identical layers.
    """
    return nn.ModuleList([module for _ in range(N)])


class Sync(nn.Module):
    """
        Paper required obj_feat and ocr_feat has the same d-dimension
        Sync to one dimension (..., 1024, 2048, ...)
        Init once and updating parameters
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.sync = nn.Linear(
            in_features=in_dim,
            out_features=out_dim
        )

    def forward(self, feats):
        """
            :params feats:   BS, num, original_feat_dim
        """
        return self.sync(feats)



#============== OCR Embedding ================
class BaseEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_config = registry.get_config("model_attributes")
        self.device = registry.get_args("device")
        self.hidden_size = self.model_config["hidden_size"]
        self.common_dim = self.model_config["feature_dim"]

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))



class SublayerConnection(nn.Module):
    """
        A residual connection followed by a layer norm.
        Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, input_feat, layer_output):
        """
            Apply residual connection to any sublayer with the same size.
            Args:
                input_feat: Features input ban đầu:
                layer_output: Output của từng layer
        """
        return input_feat + self.dropout(self.layer_norm(layer_output))



class OCREncoderLayer(BaseEmbedding):
    def __init__(self):
        super(OCREncoderLayer, self).__init__()
        self.build_config()
        self.ffn = PositionwiseFeedForward(
            d_model=self.hidden_size,
            d_ff=3072 # ViT5 d_ff config
        )
        self.self_attn = ScaledDotProductAttention(
            num_heads=self.self_attn_config["num_heads"], 
            d_model=self.hidden_size
        )
        self.constituent_module = ViConstituentModule()
        self.layer_connection = clones(
            module=SublayerConnection(size=self.hidden_size),
            N=2
        )
        
    def build_config(self):
        self.ocr_config = self.model_config["ocr"]
        self.self_attn_config = self.ocr_config["self_attn"]

    def forward(self, ocr_features, ocr_mask, attn_gate):
        """
            Args:
                ocr_features: features for ocr tokens
                ocr_mask: Mask for tokens
                attn_gate: Show the attention scores of connection of 1 token to others token
        """
        # attn_gate: Xác suất nhóm tích lũy với toàn bộ token
        # neibor_attn: Xác suất ảnh hưởng đối với neighbor liền kề only
        attn_gate, neibor_attn = self.constituent_module(ocr_features, ocr_mask, attn_gate)
        semantic_ocr_features = self.self_attn(
            queries=ocr_features,
            keys=ocr_features,
            values=ocr_features,
            attn_gate=attn_gate,
            attn_mask=ocr_mask
        )
        #-- Smooth and enhance semantic embedding
        semantic_ocr_features = self.layer_connection[0](input_feat=ocr_features, layer_output=semantic_ocr_features)
        semantic_ocr_ffn_features = self.ffn(semantic_ocr_features)
        semantic_ocr_features = self.layer_connection[1](input_feat=semantic_ocr_features, layer_output=semantic_ocr_ffn_features)
        return semantic_ocr_features, attn_gate, neibor_attn



class OCREncoder(BaseEmbedding):
    def __init__(self):
        super(OCREncoder, self).__init__()
        self.layers = clones(
            module=OCREncoderLayer(), 
            N=3
        )
        self.layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, ocr_features, ocr_mask):
        attn_gate = 0. #-- Init 0 at fist, when pass through OCREncoderLayer -> Turn to tensor BS, 1, M, M with head
        stack_neibor_attn = [] #-- To visualize impact of word to its neighbour 
        semantic_ocr_features = ocr_features
        for sub_layer in self.layers:
            semantic_ocr_features, attn_gate, neibor_attn = sub_layer(
                ocr_features=semantic_ocr_features, 
                ocr_mask=ocr_mask, 
                attn_gate=attn_gate
            )
            stack_neibor_attn.append(neibor_attn)
        stack_neibor_attn = torch.concat(stack_neibor_attn, dim=1)
        return semantic_ocr_features, stack_neibor_attn


#============== Obj Embedding ================
class OBJEncoder(BaseEmbedding):
    def __init__(self):
        super().__init__()
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

        return self.dropout(self.gelu(linear_box_out + linear_feat_out))

     


