#!/bin/bash

# Imposta i valori predefiniti per i flag
dataset=""
finetune="false"

# Analizza gli argomenti dalla linea di comando utilizzando getopts
while getopts ":d:f:m:" opt; do
  case $opt in
    d)
      dataset="$OPTARG"
      ;;
    f)
      finetune="$OPTARG"
      ;;
    \?)
      echo "Opzione non valida: -$OPTARG" >&2
      ;;
    :)
      echo "L'opzione -$OPTARG richiede un argomento." >&2
      ;;
  esac
done

# Stampa i valori dei parametri
echo "Dataset: $dataset"
echo "Finetune: $finetune"

if [ "$finetune" = "true" ]; then
    eval_type="finetune"
    echo "Finetune"
else
    eval_type="linear"
    echo "Linear"
fi


# Directory contenente i file YAML
CONFIG_PATH="scripts/kfold/$eval_type/$dataset"

# File bash per lanciare i comandi
OUTPUT_SCRIPT="slurm_$dataset_$eval_type.sh"

# Crea o svuota il file di output
> $OUTPUT_SCRIPT

# Itera sui file YAML nella cartella CONFIG_PATH
for config_file in $CONFIG_PATH/*.yaml; do
    # Estrai il nome del file senza estensione
    config_name=$(basename "$config_file")
    # Costruisci il comando sbatch con il file di configurazione corrente
    command="sbatch slurm/slurm_eval_fixed_param --config-path $CONFIG_PATH --config-name $config_name"
    # Aggiungi il comando al file di output e eseguilo in modo asincrono
    echo "$command" >> $OUTPUT_SCRIPT
    echo "Lancio comando: $command"
    echo "$command" &
done
