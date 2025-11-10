import diffusers
import torch
import os
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2


def load_model(path, device):
    pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
        path, variant="fp16", torch_dtype=torch.float16
    ).to(device)
    return pipe


def load_image(path):
    img = cv2.imread(path,1)
    # Convert BGR (OpenCV format) to RGB
    cv2_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    img = Image.fromarray(cv2_rgb)
    return img

#----------DATASET----------
class ViInforgraphicDataset(Dataset):
    def __init__(self, image_dir):
        super().__init__()
        list_image_names = os.listdir(image_dir)
        self.data = [
            os.path.join(image_dir, name)
            for name in list_image_names
        ]

    def __getitem__(self, idx):
        return {
            "id": self.data[idx].split("/")[-1].split(".")[0],
            "image_path": self.data[idx]
        }
    
    def __len__(self):
        return len(self.data)
        

def collate_fn(batch):
    # ic(batch)
    list_id = [item["id"] for item in batch]
    list_image_path = [item["image_path"] for item in batch]
    return list_id, list_image_path


def get_loader(img_dir):
    dataset = ViInforgraphicDataset(img_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    return dataloader


if __name__=="__main__":
    image_dir = "/datastore/npl/ViInfographicCaps/data/images"
    path = "/datastore/npl/ViInfographicCaps/model/marigold-depth-v1-0"
    save_dir = "/datastore/npl/ViInfographicCaps/data/depth_images"
    
    if torch.cuda.is_available():
        print("Exist cuda")
    else:
        raise Exception("No cuda found")

    device = "cuda:0"
    pipe = load_model(path, device=device)
    dataloader = get_loader(image_dir)

    for list_id, list_image_path in tqdm(dataloader, desc="Extracting depth estimation"):
        # ic(list_image_path)
        list_id = [
            id
            for id in list_id
            if not os.path.exists(os.path.join(save_dir, f"{id}.png"))
        ]
        if len(list_id)==0:
            continue
        
        images = []
        for img_path in list_image_path:
            try:
                image = diffusers.utils.load_image(img_path)
            except:
                image = load_image(img_path)
            images.append(image)

        depth = pipe(images)
        depth_16bit = pipe.image_processor.export_depth_to_16bit_png(depth.prediction)
        for id, depth in zip(list_id, depth_16bit):
            save_path = os.path.join(save_dir, f"{id}.png")
            depth.save(save_path)

