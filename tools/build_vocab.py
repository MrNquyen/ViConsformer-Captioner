from tqdm import tqdm
from collections import Counter
import sys
import os
sys.path.append(os.getcwd())
from utils.utils import load_json, save_vocab

TRAIN_PATH="./data/train.json"
SAVE_PATH = "./data/vocab.txt"
# VAL_PATH="./data/val.json"
# TEST_PATH="./data/test.json"

if __name__=="__main__":
    train_json = load_json(TRAIN_PATH)
    captions = [v["caption"] for k, v in tqdm(train_json.items())]
    vocabs = []
    for caption in captions:
        tokens = caption.split(" ")
        vocabs.extend(tokens)
    counter = Counter(vocabs)
    counter = {
        k: count
        for k, count in counter.items()
        if count > 5
    }
    vocabs = counter.keys()
    save_vocab(vocabs, SAVE_PATH)