import wandb
import yaml
import os
import argparse


def get_wandb_config(run_id):
    # Inizializza W&B con il tuo API key
    api = wandb.Api()
    run = api.run(f"thomas-cecconello/solo-learn/{run_id}")
    config = run.config

    pretrain_method = config["method"]
    name = config["name"]
    epochs = config["max_epochs"]  # Assicurati che questo campo esista nella tua configurazione
    backbone = config["backbone"]["name"]  # Assicurati che questo campo esista nella tua configurazione
    
    wandb.finish()  # Chiudi l'esperimento W&B

    return pretrain_method, name, epochs, backbone

def generate_pretrained_feature_extractor_path(pretrain_method, run_id, name, epochs):
    return f"trained_models/{pretrain_method}/{run_id}/{name}-{run_id}-ep={epochs-1}.ckpt"

def generate_yaml_from_wandb(run_id, dataset, dataset_path, devices, num_workers, K_fold_value, exp_repetitions, datalist, balancing_strategy, sample_size):
    pretrain_method, pretrain_name, epochs, backbone = get_wandb_config(run_id)
    pretrained_feature_extractor = generate_pretrained_feature_extractor_path(pretrain_method, run_id, pretrain_name, epochs)
    
    name = f"__linear__{dataset}__minmax__{pretrain_name}__K{K_fold_value}__{balancing_strategy}__{os.path.splitext(datalist)[0]}__"
    data_path = dataset_path
    #"../2-ROBIN" if dataset == "robin" else "../RGZ-D1-smorph-dataset"
    
    config = {
        "defaults": [
            "_self_",
            {"wandb": "private.yaml"},
            {"augmentations": "asymmetric_minmax.yaml"},
            {"override hydra/hydra_logging": "disabled"},
            {"override hydra/job_logging": "disabled"}
        ],
        "hydra": {
            "output_subdir": None,
            "run": {
                "dir": "."
            }
        },
        "name": name,
        "pretrained_feature_extractor": pretrained_feature_extractor,
        "backbone": {
            "name": backbone
        },
        "pretrain_method": pretrain_method,
        "data": {
            "dataset": dataset,
            "train_path": data_path,
            "val_path": data_path,
            "datalist": datalist,
            "balancing_strategy": balancing_strategy,
            "sample_size": sample_size,
            "format": "image_folder",
            "num_workers": num_workers
        },
        "optimizer": {
            "name": "sgd",
            "batch_size": 256,
            "lr": 0.3,
            "weight_decay": 0
        },
        "scheduler": {
            "name": "step",
            "lr_decay_steps": [10, 80]
        },
        "checkpoint": {
            "enabled": False,
            "dir": "trained_models",
            "frequency": 25
        },
        "auto_resume": {
            "enabled": False
        },
        "performance": {
            "disable_channel_last": True
        },
        "max_epochs": 100,
        "devices": devices,
        "sync_batchnorm": True,
        "accelerator": "gpu",
        "strategy": "auto",
        "precision": 16,
        "K_fold": K_fold_value,
        "repetitions": exp_repetitions
    }
    
    yaml_content = yaml.dump(config)

    output_directory = os.path.join("scripts", "kfold", "linear", dataset)
    yaml_filename = f"{name}.yaml"
    yaml_filepath = os.path.join(output_directory, yaml_filename)

    with open(yaml_filepath, "w") as yaml_file:
        yaml_file.write(yaml_content)

    name = f"__linear__{dataset}__mixed__{pretrain_name}__K{K_fold_value}__{balancing_strategy}__{os.path.splitext(datalist)[0]}__"

    config = {
        "defaults": [
            "_self_",
            {"wandb": "private.yaml"},
            {"augmentations": "asymmetric_mixed.yaml"},
            {"override hydra/hydra_logging": "disabled"},
            {"override hydra/job_logging": "disabled"}
        ],
        "hydra": {
            "output_subdir": None,
            "run": {
                "dir": "."
            }
        },
        "name": name,
        "pretrained_feature_extractor": pretrained_feature_extractor,
        "backbone": {
            "name": backbone
        },
        "pretrain_method": pretrain_method,
        "data": {
            "dataset": dataset,
            "train_path": data_path,
            "val_path": data_path,
            "datalist": datalist,
            "balancing_strategy": balancing_strategy,
            "sample_size": sample_size,
            "format": "image_folder",
            "num_workers": num_workers
        },
        "optimizer": {
            "name": "sgd",
            "batch_size": 256,
            "lr": 0.3,
            "weight_decay": 0
        },
        "scheduler": {
            "name": "step",
            "lr_decay_steps": [10, 80]
        },
        "checkpoint": {
            "enabled": False,
            "dir": "trained_models",
            "frequency": 25
        },
        "auto_resume": {
            "enabled": False
        },
        "performance": {
            "disable_channel_last": True
        },
        "max_epochs": 100,
        "devices": devices,
        "sync_batchnorm": True,
        "accelerator": "gpu",
        "strategy": "auto",
        "precision": 16,
        "K_fold": K_fold_value,
        "repetitions": exp_repetitions
    }
    
    yaml_content = yaml.dump(config)

    output_directory = os.path.join("scripts", "kfold", "linear", dataset)
    yaml_filename = f"{name}.yaml"
    yaml_filepath = os.path.join(output_directory, yaml_filename)

    with open(yaml_filepath, "w") as yaml_file:
        yaml_file.write(yaml_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera file YAML per la validazione di modelli addestrati.")
    parser.add_argument("--run_id", required=True, help="ID dell'esperimento su W&B")
    parser.add_argument("--devices", nargs="+", type=int, required=True, help="Lista di dispositivi")
    parser.add_argument("--num_workers", type=int, default=8, help="Numero di worker per il dataloader")
    parser.add_argument("--K", type=int, default=10, help="Numero di fold per la K-cross-fold validation")
    parser.add_argument("--reps", type=int, default=1, help="Numero di ripetizioni dell'esperimento")
    parser.add_argument("--dataset", type=str, default="robin", help="Nome del dataset")
    parser.add_argument("--dataset_path", type=str, default="../2-ROBIN", help="Path del dataset")
    parser.add_argument("--datalist", type=str, default="info.json", help="Nome datalist in formato .json")
    parser.add_argument("--balancing_strategy", type=str, default="as_is", help="as_is, balanced_downsampled,  balanced_fixed")
    parser.add_argument("--sample_size", type=int, default=1650, help="Numero di esempi per classe in caso di bilanciamento fixed")


    args = parser.parse_args()

    #datasets = ["rgz", "robin"]  # Aggiungi altri dataset se necessario

    #for dataset in datasets:
    generate_yaml_from_wandb(args.run_id, args.dataset, args.dataset_path, args.devices, args.num_workers, args.K, args.reps, args.datalist, args.balancing_strategy, args.sample_size)

    # python generate_eval_experiments.py --run_id uja9qvb7 --devices 0 1
    # python generate_eval_experiments.py --run_id uja9qvb7 --devices 0 1 --num_workers 16
