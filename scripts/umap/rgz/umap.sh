python3 main_umap.py \
    --dataset rgz \
    --train_data_path "../RGZ-D1-smorph-dataset" \
    --val_data_path "../RGZ-D1-smorph-dataset" \
    --batch_size 16 \
    --num_workers 4 \
    --pretrained_checkpoint_dir "./trained_models/byol/ddn5f8w5/"
