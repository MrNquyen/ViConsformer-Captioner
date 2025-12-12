# Implementation for [ViConsFormer: Constituting Meaningful Phrases of Scene Texts using Transformer-based Method in Vietnamese Text-based Visual Question Answering]([https://arxiv.org/abs/2302.01540](https://aclanthology.org/2024.paclic-1.75/))

## **Data Preparation**
- Object Features: Extract YOLO on OpenImageV7
- OCR Features: Extract SwinTextSpotter

## Update Modules
```
  Completed
```

## Setup
Setup model using
```
  ./scripts/setup.sh
```

Or:
```
  python setup.py build_ext --inplace
```

## Training
Training on your own dataset:
```
  ./scripts/train.sh
```

Or:
```
  python main.py \
    --config ./config/config.yaml \
    --save_dir ./save \
    --run_type train \
    --device 7
```

## Testing 
Testing on trained model:
```
  ./scripts/test.sh
```

Or:
```
  python tools/run.py \
    --config ./config/config.yaml \
    --save_dir ./save \
    --run_type inference\
    --device 7
    --resume_file your/ckpt/path
```


