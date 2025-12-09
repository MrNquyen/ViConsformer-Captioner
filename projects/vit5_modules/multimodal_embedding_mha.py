
import torch
from torch import nn
from typing import List

from tqdm import tqdm
from time import time
from icecream import ic

from vit5_modules.scene_text_embedding import SceneTextEmbedding
from vit5_modules.attention import SpartialCirclePosition, ScaledDotProductAttention
from vit5_modules.image_embedding import ImageEmbedding


#----------UTILS----------
def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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


# ================ OCREmbedding =============================
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



class OCRTokenEncoder(BaseEmbedding):
    def __init__(self):
        pass

    def forward(self):
        pass



class OCRTokenEncoderLayers(BaseEmbedding):
    def __init__(self):
        pass

    def forward(self):
        """
            Layer 
            Contains:
                - GroupAttention            : GroupAttention is constituate modules (using multihead)
                - ScaledDotProductAttention : Attention Between Q, K, V adding group probs (using multihead)
                - PositionwiseFeedForward   : Position Embedding
                - SublayerConnection        : Residual Connection through Layer
        """
        pass

        

# ================ ObjEmbedding =============================
class OBJEmbedding(BaseEmbedding):
    def __init__(self):
        super().__init__()
        self.image_embedding = ImageEmbedding()

    def forward(self):
        obj_features = self.image_embedding(batch)
        obj_spatial_att_embed = self.spatial_circle_position(
            batch=batch, 
            features=obj_features,
            list_boxes=batch["list_obj_boxes"],
            features_mask=batch["obj_mask"]
        )
        return obj_spatial_att_embed

