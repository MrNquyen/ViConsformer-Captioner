import torch
from torch import nn
from typing import List

from tqdm import tqdm
from time import time
from icecream import ic

from vit5_modules.scene_text_embedding import SceneTextEmbedding
from vit5_modules.spatial_circle_position import SpartialCirclePosition
from vit5_modules.image_embedding import ImageEmbedding


#----------SYNC INPUT DIM----------
class Sync(nn.Module):
    # Paper required obj_feat and ocr_feat has the same d-dimension
    # Sync to one dimension (..., 1024, 2048, ...)
    # Init once and updating parameters
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


#----------Word Tokenizer----------
class WordTokenizer:
    def __init__(self, tokenizer):
        pass

    


#----------Embedding----------
class BaseEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = config
        self.device = device
        self.hidden_size = config["hidden_size"]
        self.common_dim = config["feature_dim"]

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)



class OCREmbedding(BaseEmbedding):
    def __init__(self, encoder_ocr_tokens):
        super().__init__()
        self.encoder_ocr_tokens = encoder_ocr_tokens
        self.spatial_circle_position = SpartialCirclePosition()
        self.scene_text_embedding = SceneTextEmbedding()


    def ocr_token_embedding(self, batch):
        list_ocr_tokens_string = [" <context> ".join(ocr_tokens) for ocr_tokens in batch["list_ocr_tokens"]]
        ocr_tokens_inputs_vit5 = self.encoder_ocr_tokens.tokenize(list_ocr_tokens_string)        
        ocr_tokens_embed_vit5 = self.encoder_ocr_tokens.text_embedding(ocr_tokens_inputs_vit5)
        return ocr_tokens_embed_vit5
        

    def forward(self, batch):
        ocr_tokens_embed_vit5   = self.ocr_token_embedding(batch)
        scene_text_embed        = self.scene_text_embedding(batch, ocr_tokens_embed_vit5)
        ocr_spatial_att_embed   = self.spatial_circle_position(
            batch=batch, 
            features=scene_text_embed,
            list_boxes=batch["list_ocr_boxes"],
            features_mask=batch["ocr_mask"],
        )
        return ocr_spatial_att_embed
        


class OBJEncoder(BaseEmbedding):
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

        return self.dropout(self.gelu(linear_box_out + linear_feat_out))

