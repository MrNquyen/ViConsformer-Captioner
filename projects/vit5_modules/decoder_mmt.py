import torch
import math
from torch import nn
from torch.nn import functional as F
from utils.registry import registry 
from icecream import ic

from utils.module_utils import _batch_gather, _get_causal_mask


class Decoder(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.config = registry.get_config("model_attributes")
        self.hidden_size = self.config["hidden_size"]
        self.decoder = decoder

        self.build_layer()


    def build_layer(self):
        fasttext_dim = 300
        self.ocr_semantic_linear = nn.Linear(
            in_features=fasttext_dim,
            out_features=self.hidden_size
        )

    def _shift_right(self, x):
        return self.decoder._shift_right(x)
    

    def forward(
        self,
        obj_embed,
        obj_mask,
        ocr_embed,
        ocr_mask,
        ocr_tokens_embed_vit5,
        ocr_tokens_attention_mask_vit5,
        semantic_ocr_tokens_feat,
        visual_object_concept_feat,
        decoder_input_ids,
        decoder_attention_mask
    ):        
        #-- Encoder Input Embedded
        semantic_ocr_tokens_feat = self.ocr_semantic_linear(semantic_ocr_tokens_feat)
        input_embed = torch.cat(
            [obj_embed, ocr_embed, semantic_ocr_tokens_feat, ocr_tokens_embed_vit5, visual_object_concept_feat],
            dim=1
        )

        #-- Attention mask
        visual_concept_mask = torch.ones(
            visual_object_concept_feat.size(0),
            visual_object_concept_feat.size(1),
            dtype=torch.float32,
            device=visual_object_concept_feat.device
        )
        
        input_attention_mask = torch.cat(
            [obj_mask, ocr_mask, ocr_mask, ocr_tokens_attention_mask_vit5, visual_concept_mask],
            dim=1
        )

        #-- Decoder
        vit5_dec_output = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=input_embed,
            encoder_attention_mask=input_attention_mask,
            return_dict=True,
        )

        vit5_dec_last_hidden_state = vit5_dec_output.last_hidden_state
        results = {
            "vit5_dec_last_hidden_state": vit5_dec_last_hidden_state
        }

        return results

    