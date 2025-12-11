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

    def _shift_right(self, x):
        return self.decoder._shift_right(x)
    

    def forward(
        self,
        encoder_output_embed,
        encoder_output_mask,
        decoder_input_ids,
        decoder_attention_mask
    ):        
        #-- Decoder
        vit5_dec_output = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_output_embed,
            encoder_attention_mask=encoder_output_mask,
            return_dict=True,
        )

        vit5_dec_last_hidden_state = vit5_dec_output.last_hidden_state
        results = {
            "vit5_dec_last_hidden_state": vit5_dec_last_hidden_state
        }

        return results

    