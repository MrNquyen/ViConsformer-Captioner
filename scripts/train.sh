# ViT5 Training
python main_vit5_epoch.py \
--config ./config/config.yaml \
--save_dir ./save \
--run_type train \
--device cuda:2 \
--resume_file /datastore/npl/ViInfographicCaps/workspace/baseline/ViT5-DEVICE-Image-Captioning/save/checkpoints/model_last.pth
