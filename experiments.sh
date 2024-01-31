
python3 main_kfold_crossval.py --config-path scripts/kfold/linear/robin --config-name __linear__robin__minmax__byol_banner_aug_mixed_model_resnet18__K10__balanced_fixed__info__.yaml
python3 main_kfold_crossval.py --config-path scripts/kfold/linear/robin --config-name __linear__robin__minmax__byol_hulk_aug_minmax_model_resnet18__K10__balanced_fixed__info__.yaml
python3 main_kfold_crossval.py --config-path scripts/kfold/linear/robin --config-name __linear__robin__minmax__byol_hulk_aug_mixed_model_resnet18__K10__balanced_fixed__info__.yaml
python3 main_kfold_crossval.py --config-path scripts/kfold/linear/robin --config-name __linear__robin__minmax__simclr_banner_aug_minmax_model_resnet18__K10__balanced_fixed__info__.yaml
python3 main_kfold_crossval.py --config-path scripts/kfold/linear/robin --config-name __linear__robin__minmax__simclr_banner_aug_mixed_model_resnet18__K10__balanced_fixed__info__.yaml
python3 main_kfold_crossval.py --config-path scripts/kfold/linear/robin --config-name __linear__robin__mixed__byol_banner_aug_mixed_model_resnet18__K10__balanced_fixed__info__.yaml
python3 main_kfold_crossval.py --config-path scripts/kfold/linear/robin --config-name __linear__robin__mixed__byol_hulk_aug_minmax_model_resnet18__K10__balanced_fixed__info__.yaml
python3 main_kfold_crossval.py --config-path scripts/kfold/linear/robin --config-name __linear__robin__mixed__byol_hulk_aug_mixed_model_resnet18__K10__balanced_fixed__info__.yaml
python3 main_kfold_crossval.py --config-path scripts/kfold/linear/robin --config-name __linear__robin__mixed__simclr_banner_aug_minmax_model_resnet18__K10__balanced_fixed__info__.yaml
python3 main_kfold_crossval.py --config-path scripts/kfold/linear/robin --config-name __linear__robin__mixed__simclr_banner_aug_mixed_model_resnet18__K10__balanced_fixed__info__.yaml

: '
python3 main_linear.py --config-path scripts/finetune/robin --config-name byol_banner_minmax.yaml
python3 main_linear.py --config-path scripts/finetune/robin --config-name byol_banner_mixed.yaml
python3 main_linear.py --config-path scripts/finetune/robin --config-name byol_imagenet_minmax.yaml
python3 main_linear.py --config-path scripts/finetune/robin --config-name byol_imagenet_mixed.yaml
python3 main_linear.py --config-path scripts/finetune/robin --config-name mae_banner_minmax.yaml
python3 main_linear.py --config-path scripts/finetune/robin --config-name mae_banner_mixed.yaml


python3 main_linear.py --config-path scripts/linear/rgz --config-name byol_banner_minmax.yaml
python3 main_linear.py --config-path scripts/linear/rgz --config-name byol_banner_mixed.yaml
python3 main_linear.py --config-path scripts/linear/rgz --config-name byol_imagenet_minmax.yaml
python3 main_linear.py --config-path scripts/linear/rgz --config-name byol_imagenet_mixed.yaml
python3 main_linear.py --config-path scripts/linear/rgz --config-name mae_banner_minmax.yaml
python3 main_linear.py --config-path scripts/linear/rgz --config-name mae_banner_mixed.yaml

python3 main_linear.py --config-path scripts/linear/robin --config-name byol_banner_minmax.yaml
python3 main_linear.py --config-path scripts/linear/robin --config-name byol_banner_mixed.yaml
python3 main_linear.py --config-path scripts/linear/robin --config-name byol_imagenet_minmax.yaml
python3 main_linear.py --config-path scripts/linear/robin --config-name byol_imagenet_mixed.yaml
python3 main_linear.py --config-path scripts/linear/robin --config-name mae_banner_minmax.yaml
python3 main_linear.py --config-path scripts/linear/robin --config-name mae_banner_mixed.yaml

python3 main_linear.py --config-path scripts/finetune/rgz --config-name byol_banner_minmax.yaml
python3 main_linear.py --config-path scripts/finetune/rgz --config-name byol_banner_mixed.yaml
python3 main_linear.py --config-path scripts/finetune/rgz --config-name byol_imagenet_minmax.yaml
python3 main_linear.py --config-path scripts/finetune/rgz --config-name byol_imagenet_mixed.yaml
python3 main_linear.py --config-path scripts/finetune/rgz --config-name mae_banner_minmax.yaml
python3 main_linear.py --config-path scripts/finetune/rgz --config-name mae_banner_mixed.yaml

python3 main_linear.py --config-path scripts/finetune/robin --config-name byol_banner_minmax.yaml
python3 main_linear.py --config-path scripts/finetune/robin --config-name byol_banner_mixed.yaml
python3 main_linear.py --config-path scripts/finetune/robin --config-name byol_imagenet_minmax.yaml
python3 main_linear.py --config-path scripts/finetune/robin --config-name byol_imagenet_mixed.yaml
python3 main_linear.py --config-path scripts/finetune/robin --config-name mae_banner_minmax.yaml
python3 main_linear.py --config-path scripts/finetune/robin --config-name mae_banner_mixed.yaml
'