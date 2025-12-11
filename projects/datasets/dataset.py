import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from utils.utils import load_json, load_npy
from utils.transform import Transform
from icecream import ic

#----------DATASET----------
class ViInforgraphicDataset(Dataset):
    def __init__(self, dataset_config, split):
        super().__init__()

        #-- Load features directory
        ocr_feat_dir, obj_feat_dir = dataset_config["image_features"][split].split(", ")
        imdb_path = dataset_config["imdb_files"][split]

        #-- Load data
        imdb = load_npy(imdb_path)
        self.data = []
        self.transform = Transform(image_size=448).transform_from_ndarray()
        for item in tqdm(imdb, desc=f"Loading {split} split"):
            im_id = item["image_id"]

            #-- Load features path
            ocr_feat_path = os.path.join(ocr_feat_dir, f"{im_id}.npy")
            obj_feat_path = os.path.join(obj_feat_dir, f"{im_id}.npy")
            
            #-- Load features            
            ocr_feat = load_npy(ocr_feat_path)
            obj_feat = load_npy(obj_feat_path)
            
            #-- Load data
            self.data.append({
                "id": im_id,
                "im_path": item["image_path"],
                "im_width": item["image_width"],
                "im_height": item["image_height"],
                "ocr_tokens": item["ocr_tokens"],
                "ocr_boxes": item["ocr_normalized_boxes"],
                "obj_boxes": item["obj_normalized_boxes"],
                "ocr_scores": item["ocr_scores"],
                "caption_tokens": item["caption_tokens"],
                "ocr_feat": ocr_feat.item()["det_features"],
                "obj_feat": obj_feat,
                "caption_str": item["caption_str"],
            })


    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    list_id = [item["id"] for item in batch]
    list_im_path = [item["im_path"] for item in batch]
    list_im_width = [item["im_width"] for item in batch]
    list_im_height = [item["im_height"] for item in batch]
    list_ocr_tokens = [item["ocr_tokens"] for item in batch]
    list_ocr_boxes = [item["ocr_boxes"] for item in batch]    
    list_obj_boxes = [item["obj_boxes"] for item in batch]
    list_ocr_scores = [item["ocr_scores"] for item in batch]
    list_caption_tokens = [item["caption_tokens"] for item in batch]
    list_ocr_feat = [item["ocr_feat"] for item in batch]
    list_obj_feat = [item["obj_feat"] for item in batch]
    list_captions = [item["caption_str"] for item in batch]

    return {
        "list_id": list_id,
        "list_im_path": list_im_path,
        "list_im_width": list_im_width,
        "list_im_height": list_im_height,
        "list_ocr_tokens": list_ocr_tokens,
        "list_ocr_boxes": list_ocr_boxes,
        "list_obj_boxes": list_obj_boxes,
        "list_ocr_scores": list_ocr_scores,
        "list_caption_tokens": list_caption_tokens,
        "list_ocr_feat": list_ocr_feat,
        "list_obj_feat": list_obj_feat,
        "list_captions": list_captions
    }


def get_loader(dataset_config, batch_size, split):
    if split not in ["train", "val", "test"]:
        raise ValueError(f"No split found for {split}")
    dataset = ViInforgraphicDataset(dataset_config, split)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=split=="train",
        collate_fn=collate_fn,
    )
    return dataloader
