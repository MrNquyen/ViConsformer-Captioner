# M4C Training
python main.py \
--config ./config/device_config.yaml \
--save_dir ./save \
--run_type train \
--device cuda:4 \
--resume_file /datastore/npl/ViInfographicCaps/workspace/baseline/Refactor-DEVICE/DEVICE-Image-Captioning/save/checkpoints/model_last.pth


# ViT5 Training
python main_vit5_epoch.py \
--config ./config/device_config.yaml \
--save_dir ./save \
--run_type train \
--device cuda:7 \
--resume_file /datastore/npl/ViInfographicCaps/workspace/baseline/ViT5-DEVICE-Image-Captioning/save/checkpoints/model_last.pth
