import json
import numpy as np
import yaml
import cv2
import torch
import random

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import functools
import os

#---- Load json
def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        json_content = json.load(file)
        return json_content
    
#---- Save json
def save_json(path, content):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=3)


#---- Load numpy
def load_npy(path):
    return np.load(path, allow_pickle=True)


#---- Save vocab
def save_vocab(list_vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for word in list_vocab:
            f.write(f"{word}\n")



#---- Load vocab
def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]
        return vocab

#---- Get name of the image
def get_img_name(name):
    return name.split(".")[0]

#---- Load yaml file
def load_yml(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        return config
    
#---- Load yaml file
def load_img(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

@functools.lru_cache(maxsize=256)  # adjust size as needed
def load_img_cache(path):
    # Read binary first
    with open(path, 'rb') as f:
        img_bytes = f.read()
    # Convert to NumPy array and decode
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

def load_list_images_fast(image_paths, num_workers=8, desc="Loading images"):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        images = list(tqdm(executor.map(load_img_cache, image_paths), total=len(image_paths), desc=desc))
    return images


#---- Check where nan
def count_nan(tensor):
    nan_mask = torch.isnan(tensor)  # Boolean mask where values are NaN
    nan_count = nan_mask.sum().item()
    # nan_indices = nan_mask.nonzero(as_tuple=False)
    return nan_count

def check_requires_grad(module, name="module"):
    trainable, frozen = 0, 0
    for n, p in module.named_parameters():
        if p.requires_grad:
            print(f"[Trainable] {name}.{n}: {tuple(p.shape)}")
            trainable += 1
        else:
            print(f"[Frozen]    {name}.{n}: {tuple(p.shape)}")
            frozen += 1
    print(f"\nSummary for {name}: {trainable} trainable / {frozen} frozen parameters\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False