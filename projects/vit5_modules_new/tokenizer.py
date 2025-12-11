import torch

from typing import List
from torch import nn
from utils.registry import registry

class WordTokenizer:
    def __init__(self, tokenizer):
        self.config = registry.get_config("model_attributes")
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")
        self.max_length = self.config["max_length"]
        self.tokenizer = tokenizer

    def tokenize(
        self, 
        texts: List[str], 
        max_length: int = None,
        to_batch_max_length: bool = False,
    ):
        """
            Args:
                - texts: (str): Batch of texts

            Return:
                - dict: Text input has 'input_ids', 'attention_mask'

            Example: 
                - Return a dict = {
                    'input_ids': ..., 
                    'attention_mask': ...,
                }
        """
        if not to_batch_max_length:
            if not max_length:
                max_length = self.max_length
            inputs = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"        # return PyTorch tensors directly
            )
        else:
            inputs = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                return_tensors="pt"        # return PyTorch tensors directly
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs # 'input_ids', 'token_type_ids', 'attention_mask'
    
    #-- Decode
    def batch_decode(self, pred_inds):
        return self.tokenizer.batch_decode(pred_inds, skip_special_tokens=True)

    #-- Common function
    def get_vocab_size(self):
        return len(self.tokenizer.get_vocab())

    #-- Get token
    def get_pad_token(self):
        return self.tokenizer.pad_token
    
    def get_unk_token(self):
        return self.tokenizer.unk_token
    
    def get_cls_token(self):
        return self.tokenizer.cls_token
    
    def get_eos_token(self):
        return self.tokenizer.eos_token
    
    #-- Get token id
    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id
    
    def get_unk_token_id(self):
        return self.tokenizer.unk_token_id
    
    def get_cls_token_id(self):
        return self.tokenizer.cls_token_id
    
    def get_eos_token_id(self):
        return self.tokenizer.eos_token_id

