# Implementation for [DEVICE: Depth and Visual Concepts Aware Transformer for OCR-based Image Captioning](https://arxiv.org/abs/2302.01540)

## **Data Preparation**
- Object Features: Extract FasterRCNN
- OCR Features: Extract SwinTextSpotter
- Depth Estimation: Extract via [marigold-depth-v1-0](https://huggingface.co/prs-eth/marigold-depth-v1-0)

## Update Modules
```
  Completed
```

## Depth Estimation
Using marigold-depth-v1-0
```
  python ./tools/extract_depth.py
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
    --config ./config/device_config.yaml \
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
    --config ./config/device_config.yaml \
    --save_dir ./save \
    --run_type inference\
    --device 7
    --resume_file your/ckpt/path
```


