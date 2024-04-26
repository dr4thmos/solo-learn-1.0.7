# Imposta i valori predefiniti per i flag
dataset=""
finetune="false"
model_path="model_list.txt"

# Analizza gli argomenti dalla linea di comando utilizzando getopts
while getopts ":d:f:m:" opt; do
  case $opt in
    d)
      dataset="$OPTARG"
      ;;
    f)
      finetune="$OPTARG"
      ;;
    m)
      model_path="$OPTARG"
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
echo "Model Path: $model_path"

# Esegui un'azione basata sul valore del flag
if [ "$finetune" = "true" ]; then
    python fill_experiments_to_eval.py --dataset $dataset --finetune --model_list_path $model_path
    eval_type="finetune"
    echo "Finetune"
else
    python fill_experiments_to_eval.py --dataset $dataset --model_list_path $model_path
    eval_type="linear"
    echo "Linear"
fi

# Creazione configurazione esperimenti
sh "gen_eval_exp_$dataset.sh"
# Creazione linee di comando lanciare i job slurm per ogni esperimento
sh "create_slurm_launcher.sh" -d $dataset -f $finetune
# Lancio dei job slurm per ogni esperimento
sh "slurm_$dataset_$eval_type.sh"