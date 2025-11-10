import torch
import fasttext
from torch import nn
from typing import List

from projects.modules.convert_depth_map import DepthExtractor, DepthExtractorDirect
from projects.modules.depth_enhance_update import DeFUM
from projects.modules.semantic_guide_alignment import SgAM
from utils.module_utils import fasttext_embedding_module, _batch_padding_string
from utils.phoc.build_phoc_v2 import build_phoc
from utils.registry import registry
from utils.vocab import PretrainedVocab, OCRVocab
from tqdm import tqdm
from time import time
from icecream import ic


#----------Word embedding----------
class WordEmbedding(nn.Module):
    def __init__(self, model, tokenizer, text_embedding_config):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = text_embedding_config
        self.max_length = self.config["max_length"]

        vocab_path =self.config["common_vocab"]
        self.common_vocab = PretrainedVocab(
            model=self.model,
            tokenizer=self.tokenizer,
            vocab_file=vocab_path
        )
        
    def get_prev_inds(self, sentences, ocr_tokens):
        """
            Use to get inds of each token of the caption sentences

            Parameters:
            ----------
            sentences: List[str]
                - Caption of the images
            
            ocr_tokens: List[List[str]]

            Return:
            ----------
            prev_ids: Tensor:
                - All inds of all word in the sentences 
        """
        ocr_vocab_object = OCRVocab(ocr_tokens=ocr_tokens)

        start_token = self.common_vocab.get_start_token()
        end_token = self.common_vocab.get_end_token()
        pad_token = self.common_vocab.get_pad_token()
        
        sentences_tokens = [
            sentence.split(" ")
            for sentence in sentences
        ]
        sentences_tokens = _batch_padding_string(
            sequences=sentences_tokens,
            max_length=self.config["max_length"],
            pad_value=pad_token,
            return_mask=False
        )
        sentences_tokens = [
            [start_token] + sentence_tokens[:self.max_length - 2] + [end_token]
            for sentence_tokens in sentences_tokens
        ]

        # Get prev_inds
        prev_ids = [
            [
                self.common_vocab.get_size() + ocr_vocab_object[sen_id].get_word_idx(token)
                if token in ocr_tokens[sen_id]
                # else ocr_vocab_object[sen_id].get_word_idx(token)
                else self.common_vocab.get_word_idx(token)
                for token in sentence_tokens
            ] 
            for sen_id, sentence_tokens in enumerate(sentences_tokens)
        ]
        return torch.tensor(prev_ids)
    

#----------Embedding----------
class BaseEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = registry.get_config("model_attributes")
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")

        self.hidden_size = self.config["hidden_size"]
        self.common_dim = self.config["feature_dim"]
        
        self.LayerNorm = nn.LayerNorm(normalized_shape=self.hidden_size)


#----------- OBJ EMBEDDING ----------------
class ObjEmbedding(BaseEmbedding):
    def __init__(self):
        super().__init__()
        self.linear_feat = nn.Linear(
            in_features=self.common_dim,
            out_features=self.hidden_size
        )

        self.linear_box = nn.Linear(
            in_features=5,
            out_features=self.hidden_size
        )


    def forward(self, batch):
        list_obj_boxes = batch["list_obj_boxes"]
        list_obj_feat = batch["list_obj_feat"]
        list_obj_depth_feat = batch["list_obj_depth_feat"]

        #-- Concat depth features to boxes
        obj_extended_boxes = torch.concat(
            [list_obj_boxes, list_obj_depth_feat],
            dim=-1
        )

        #-- Calculate boxes
        linear_obj_feat = self.linear_feat(list_obj_feat)
        linear_obj_boxes = self.linear_feat(obj_extended_boxes)
        
        obj_embed_features = self.LayerNorm(linear_obj_boxes) + self.LayerNorm(linear_obj_feat)
        return obj_embed_features




#----------- OCR EMBEDDING ----------------
class OCREmbedding(BaseEmbedding):
    def __init__(self, fasttext_model):
        super().__init__()

        #-- Build
        self.build_fasttext_model()
        
        #-- Layers
        self.linear_out_defum = nn.Linear(
            in_features=self.common_dim,
            out_features=self.hidden_size
        )

        fasttext_dim = 300
        self.linear_out_sgam = nn.Linear(
            in_features=fasttext_dim,
            out_features=self.hidden_size
        )
        
        # phoc_dim = 604
        phoc_dim = 1810
        self.linear_out_phoc = nn.Linear(
            in_features=phoc_dim,
            out_features=self.hidden_size
        )

        self.linear_out_ocr_boxes = nn.Linear(
            in_features=5,
            out_features=self.hidden_size
        )

        self.linear_out_ocr_conf = nn.Linear(
            in_features=1,
            out_features=self.hidden_size
        )

        self.LayerNorm = nn.LayerNorm(normalized_shape=self.hidden_size)
        
        # Modules
        self.DeFUM(batch) = DeFUM()
        self.SgAM = SgAM(
            sgam_config=self.config["sgam"],
            fasttext_model=fasttext_model,
            hidden_size=self.config["hidden_size"]
        )

    #-- BUILD
    def build_fasttext_model(self):
        self.fasttext_model = fasttext.load_model(self.config["fasttext_bin"])


    #-- Extract Features
    def phoc_embedding(self, words: List[str]):
        """
            :params words:  List of word needed to embedded
        """
        phoc_embedding = [
            # build_phoc(token=word) 
            build_phoc(word=word) 
            for word in words
        ]
        return torch.tensor(phoc_embedding).to(self.device)
    

    def fasttext_embedding(self, words: List[str]):
        """
            :params words:  List of word needed to embedded
        """
        fasttext_embedding = [
            fasttext_embedding_module(
                model=self.fasttext_model,
                word=word
            ) 
            for word in words
        ]
        return torch.tensor(fasttext_embedding).to(self.device)

    
    #-- FORWARD
    def forward(self, batch):
        #~ Load features
        list_ocr_tokens = batch["list_ocr_tokens"]
        list_ocr_boxes = batch["list_ocr_boxes"]
        list_ocr_scores = batch["list_ocr_scores"]
        list_ocr_depth_feat = batch["list_ocr_depth_feat"]
        list_clip_image_feat = batch["list_clip_image_feat"]
        list_clip_object_concepts_feat = batch["list_clip_object_concepts_feat"]

        #~ OCR Token Fasttext / PHOC Embedding / Extend Boxes with Depth Estimaiton
        ocr_token_ft_embed = torch.tensor([self.fasttext_embedding(tokens) for tokens in list_ocr_tokens]).to(self.device)
        ocr_token_phoc_embed = torch.tensor([self.phoc_embedding(tokens) for tokens in list_ocr_tokens]).to(self.device)
        ocr_extended_boxes = torch.concat(
            [list_ocr_boxes, list_ocr_depth_feat],
            dim=-1
        )

        #~ DEFUM (Find the correlation of the object with multiple ocr tokens)
        depth_aware_visual_feat = self.DeFUM(batch)

        #~ SVOCE (Salient Visual Object Concepts Extractor)

        #~ SgAM 
        