import torch
import math
from torch import nn
from vit5_modules.base import BaseEmbedding

from utils.registry import registry


class ViConstituentModule(nn.Module):
    def __init__(self):
        self.config = registry.get_config("model_attributes")
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")

    def forward(self, batch, vit5_ocr_tokens_embed, vit5_ocr_tokens_embed_mask):
        pass