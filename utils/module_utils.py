import numpy as np
import torch 
import torch.nn.functional as F
from typing import List
from icecream import ic
# Fasttext
def fasttext_embedding_module(model, word):
    ft_feat = model.get_word_vector(word)
    return ft_feat


# Encoding
def _get_mask(nums, max_num, device="cpu"):
    """
        :params nums    :   BS,     : torch tensor of list current length of the features 
        :params max_num :   integer : max length of the features
        ----
        Use to mask mismatching number of ocr_token, object, caption_tokens:
            - 0: pad
            - 1: no-pad
    """
    # arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1) # Original
    arange = torch.arange(0, max_num).unsqueeze(0).expand(len(nums), -1)
    non_pad_mask = arange.to(device).lt(nums.unsqueeze(-1))
    return non_pad_mask.type(torch.float32)

# https://github.com/ronghanghu/mmf/blob/project/m4c_captioner_pre_release/pythia/models/m4c.py#L510
def _batch_gather(x, inds):
    """
        Gather features by inds
    """
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size*length, dim)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results

# -- Casual Mask
def _get_causal_mask(seq_length, device):
    """
        seq_length = 3
        [[1., 0., 0.],
        [1., 1., 0.],
        [1., 1., 1.]]
    """
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i+1):
            mask[i, j] = 1.
    return mask


# -- Pad input
def _batch_padding_string(
        sequences, 
        max_length,
        pad_value="<pad>", 
        return_mask=True
    ):
    """
    Pads a list of lists to the maximum length with pad_value.
    If return_mask is True, also returns a mask indicating real tokens (1) vs pads (0).

    Parameters:
    ---------
        sequences: List[List[str]]
            List of lists of string need to pad
        
        max_length: int
            Max length of padding

        pad_value: str
            Pad value
        
        return_mask: bool
            Whether return mask or not
    """
    # Prepare containers
    padded = []
    mask   = []
    
    for seq in sequences:
        seq_len = len(seq)
        # 1) Pad up to max_length
        padded_seq = seq + [pad_value] * max((max_length - seq_len), 0)
        padded.append(padded_seq)
        # 2) Mask: 1 for real tokens, 0 for pads
        if return_mask:
            mask.append([1] * seq_len + [0] * (max_length - seq_len))
    
    if return_mask:
        return padded, mask
    else:
        return padded


def _batch_padding(inputs, max_length, pad_value, return_mask=True):
    """
        Input:
            - List of features with different lengths

        Output:
            - List of features with different lengths
            - Padding Mask

        Example:
            input = [
                torch.rand(2, 10),
                torch.rand(4, 10),
                torch.rand(3, 10),
            ]

            pad_input, mask = batch_padding(
                input=input,
                max_length=5,
                pad_value=torch.zeros(1, 10)
            )
            pad_input.shape, mask
            >> torch.Size([3, 5, 10]
            >> [[1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 0, 0]]
    """
    batch_size = len(inputs)
    # ic(input)
    # for id, item in enumerate(inputs):
    #     try:
    #         item_length = item.shape[0]
    #         ic(id, item_length)
    #     except:
    #         ic(item)
    input_length = torch.tensor([item.shape[0] for item in inputs])

    # Create mask
    arange = torch.arange(0, max_length).unsqueeze(0).expand(batch_size, -1)
    
    # Padding
    pad_input = []
    for item in inputs:
        pad_post = pad_value.expand(max_length-len(item), -1)
        item = torch.concat(
            [torch.tensor(item), pad_post],
            dim=0
        )
        pad_input.append(item)
    pad_input = torch.stack(pad_input)
    if return_mask:
        mask = torch.lt(arange, input_length.unsqueeze(1)).long()
        return pad_input, mask
    else:
        return pad_input