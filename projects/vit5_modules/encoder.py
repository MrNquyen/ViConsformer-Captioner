import torch
import numpy as np

from torch import nn
from typing import List

from utils.registry import registry

class Encoder(nn.Module):
    def __init__(self, tokenizer, encoder, max_length):
        super().__init__()
        self.writer = registry.get_writer("common")
        self.model_config = registry.get_config("model_attributes")
        self.device = registry.get_args("device")
        self.hidden_size = self.model_config["hidden_size"]
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.encoder = encoder


    #-- Tokenize
    def tokenize(self, texts: List[str]):
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
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"        # return PyTorch tensors directly
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs # 'input_ids', 'token_type_ids', 'attention_mask'


    def text_embedding(self, inputs):
        """
            Args:
                - inputs: (str): Input is a output from tokenizer

            Return:
                - Tensor: Tensor of text embeb features

            Example: 
                - inputs = {
                    'input_ids': ..., 
                    'token_type_ids': ..., 
                    'attention_mask': ...,
                }
        """
        if not self.training:
            with torch.no_grad():
                encoder_outputs = self.encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                return encoder_outputs.last_hidden_state
        else:
            encoder_outputs = self.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            return encoder_outputs.last_hidden_state



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


    def forward(self, texts):
        """
            Text Embedding the batch of texts provided
            Args:
                - texts: Batch of input texts

            Return:
                - torch.Tensor: text_embed of the given texts - BS, max_length, hidden_size
        """
        inputs = self.tokenize(texts)
        text_embed = self.text_embedding(inputs)
        return text_embed


    


    