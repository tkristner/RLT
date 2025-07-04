#!/bin/bash
# Script de lancement optimisé et complet pour RLT (SFT + RL)

set -e  # Arrêt sur erreur

echo "🚀 RLT Training Launcher - SFT + RL Pipeline"
echo "=================================================================="

# Vérification du répertoire
if [ ! -f "train.py" ]; then
    echo "❌ Erreur: train.py non trouvé. Lancez depuis le répertoire RLT."
    exit 1
fi

# Vérification GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Erreur: nvidia-smi non trouvé. GPU NVIDIA requis."
    exit 1
fi

# Affichage des informations GPU
echo "📊 Informations GPU :"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# La configuration est maintenant gérée via les fichiers YAML
echo "✅ Configuration chargée depuis les fichiers cfgs/."

# On s'assure que WANDB est en mode offline
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_BENCHMARK=1
export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=true

# Fonction pour trouver le dernier répertoire step-1_SFT
find_latest_sft_model() {
    local latest_dir=""
    local latest_time=0
    
    for dir in results/step-1_SFT_*/; do
        if [ -d "$dir" ]; then
            local dir_time=$(stat -c %Y "$dir" 2>/dev/null || echo 0)
            if [ "$dir_time" -gt "$latest_time" ]; then
                latest_time=$dir_time
                latest_dir=$dir
            fi
        fi
    done
    
    echo "$latest_dir"
}

# Fonction pour trouver le modèle fusionné
find_fused_model() {
    local sft_dir="$1"
    
    # Chercher un modèle fusionné dans step-1_SFT_fused
    if [ -d "results/step-1_SFT_fused" ]; then
        if [ -f "results/step-1_SFT_fused/pytorch_model.bin" ] || [ -f "results/step-1_SFT_fused/model.safetensors" ] || ls results/step-1_SFT_fused/model-*-of-*.safetensors 1> /dev/null 2>&1; then
            echo "results/step-1_SFT_fused"
            return
        fi
    fi
    
    # Sinon chercher des adaptateurs LoRA à fusionner
    if [ -n "$sft_dir" ] && [ -f "$sft_dir/adapter_model.safetensors" ]; then
        echo "$sft_dir"
        return
    fi
    
    echo ""
}

# Détection intelligente du modèle pour l'étape RL
detect_rl_model_path() {
    # Si on utilise PEFT (LoRA), utiliser les adaptateurs originaux
    local use_peft=$(python3 -c "
import yaml
with open('cfgs/run_cfg/teacher_rlt.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('use_peft', False))
" 2>/dev/null || echo "false")
    
    if [ "$use_peft" = "True" ]; then
        # Chercher les adaptateurs LoRA originaux
        local adapter_path=$(find results/ -name "adapter_model.safetensors" -type f | head -1)
        if [ -n "$adapter_path" ]; then
            dirname "$adapter_path"
        else
            echo "results/step-1_SFT_fused"
        fi
    else
        # Utiliser le modèle fusionné
        echo "results/step-1_SFT_fused"
    fi
}

# Obtenir le chemin du modèle approprié
MODEL_PATH=$(detect_rl_model_path)
echo "📍 Modèle détecté pour l'étape RL: $MODEL_PATH"

# --- ÉTAPE 1: SUPERVISED FINE-TUNING (SFT) ---
echo ""
echo "--- ÉTAPE 1: Lancement du pré-entraînement SFT ---"
echo "Objectif: Créer le modèle de base pour l'étape RL"
echo ""

# Chercher le modèle SFT existant
LATEST_SFT_DIR=$(find_latest_sft_model)
FUSED_MODEL_PATH=$(find_fused_model "$LATEST_SFT_DIR")

if [ -n "$FUSED_MODEL_PATH" ] && [ -d "$FUSED_MODEL_PATH" ]; then
    echo "✅ Modèle SFT fusionné trouvé: $FUSED_MODEL_PATH"
    echo "   Saut de l'étape SFT."
elif [ -n "$LATEST_SFT_DIR" ] && [ -f "$LATEST_SFT_DIR/adapter_model.safetensors" ]; then
    echo "✅ Adaptateurs LoRA SFT trouvés: $LATEST_SFT_DIR"
    echo "   Fusion nécessaire..."
    
    # Fusionner les adaptateurs LoRA avec le modèle de base
    echo "🔄 Fusion des adaptateurs LoRA avec le modèle de base..."
    python3 -c "
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Charger la configuration PEFT
peft_config = PeftConfig.from_pretrained('$LATEST_SFT_DIR')
base_model_name = peft_config.base_model_name_or_path

print(f'Chargement du modèle de base: {base_model_name}')
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map='cpu',  # Charger en CPU pour la fusion
    trust_remote_code=True
)

print(f'Chargement des adaptateurs LoRA: $LATEST_SFT_DIR')
model = PeftModel.from_pretrained(base_model, '$LATEST_SFT_DIR')

print('Fusion des adaptateurs...')
merged_model = model.merge_and_unload()

print('Sauvegarde du modèle fusionné...')
os.makedirs('results/step-1_SFT_fused', exist_ok=True)
merged_model.save_pretrained('results/step-1_SFT_fused', safe_serialization=True)

# Sauvegarder aussi le tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.save_pretrained('results/step-1_SFT_fused')

print('✅ Modèle fusionné sauvé dans results/step-1_SFT_fused')
"
    FUSED_MODEL_PATH="results/step-1_SFT_fused"
else
    echo "🔧 Aucun modèle SFT trouvé. Lancement de l'entraînement..."
    # On utilise la config SFT existante et on ajoute les optimisations ultimes
    python3 train.py \
        +run_cfg=teacher_sft \
        +do_sft=true \
        use_peft=true \
        load_in_4bit=false \
        +model_args.load_in_8bit=true \
        gradient_checkpointing=true \
        +fp16=false \
        bf16=true \
        tf32=false \
        max_steps=10 \
        max_seq_length=4096 \
        packing=true \
        +trainer_args.dataloader_pin_memory=true \
        +trainer_args.remove_unused_columns=false \
        +trainer_args.dataloader_num_workers=8 \
        +make_dataset_fn.num_proc=16 \
        add_text_completions=true \
        wandb_project=rl4lm_sft_96gb \
        +trainer_args.optim=paged_adamw_8bit

    # Trouver le nouveau modèle créé
    NEW_SFT_DIR=$(find_latest_sft_model)
    if [ -z "$NEW_SFT_DIR" ] || [ ! -f "$NEW_SFT_DIR/adapter_model.safetensors" ]; then
        echo "❌ ERREUR: L'entraînement SFT a échoué. Aucun adapter_model.safetensors trouvé."
        exit 1
    else
        echo "✅ Modèle SFT créé avec succès dans: $NEW_SFT_DIR"
        
        # Fusionner les adaptateurs LoRA avec le modèle de base
        echo "🔄 Fusion des adaptateurs LoRA avec le modèle de base..."
        python3 -c "
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Charger la configuration PEFT
peft_config = PeftConfig.from_pretrained('$NEW_SFT_DIR')
base_model_name = peft_config.base_model_name_or_path

print(f'Chargement du modèle de base: {base_model_name}')
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map='cpu',  # Charger en CPU pour la fusion
    trust_remote_code=True
)

print(f'Chargement des adaptateurs LoRA: $NEW_SFT_DIR')
model = PeftModel.from_pretrained(base_model, '$NEW_SFT_DIR')

print('Fusion des adaptateurs...')
merged_model = model.merge_and_unload()

print('Sauvegarde du modèle fusionné...')
os.makedirs('results/step-1_SFT_fused', exist_ok=True)
merged_model.save_pretrained('results/step-1_SFT_fused', safe_serialization=True)

# Sauvegarder aussi le tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.save_pretrained('results/step-1_SFT_fused')

print('✅ Modèle fusionné sauvé dans results/step-1_SFT_fused')
"
        FUSED_MODEL_PATH="results/step-1_SFT_fused"
    fi
fi

echo "✅ Étape 1 (SFT) terminée avec succès !"
echo "📁 Modèle SFT disponible: $FUSED_MODEL_PATH"

# --- ÉTAPE 2: REINFORCEMENT LEARNING (RL) ---
echo ""
echo "--- ÉTAPE 2: Lancement de l'entraînement par renforcement (RL) ---"

# Vérifier que le modèle fusionné existe
if [ ! -f "$FUSED_MODEL_PATH/pytorch_model.bin" ] && [ ! -f "$FUSED_MODEL_PATH/model.safetensors" ] && ! ls "$FUSED_MODEL_PATH"/model-*-of-*.safetensors 1> /dev/null 2>&1; then
    echo "❌ ERREUR: Modèle fusionné non trouvé dans $FUSED_MODEL_PATH"
    exit 1
fi

# Lancement du serveur vLLM *uniquement* si USE_EXTERNAL_VLLM=1
if [ "${USE_EXTERNAL_VLLM:-0}" = "1" ]; then
  VLLM_PORT=${VLLM_PORT:-8000}
  VLLM_QUANT=${VLLM_QUANT:-bitsandbytes}

  echo "🚀 Démarrage serveur vLLM externe (port=$VLLM_PORT, quant=$VLLM_QUANT)…"
  ./scripts/launch_vllm_server.sh "$VLLM_PORT" "$VLLM_QUANT" > vllm.log 2>&1 &
  VLLM_PID=$!

  echo -n "⌛ Attente de la disponibilité du serveur vLLM… "
  for i in {1..30}; do
    if curl -s "http://127.0.0.1:$VLLM_PORT/health/" >/dev/null; then echo "OK"; break; fi
    sleep 2
  done

  echo -n "⌛ Initialisation du moteur vLLM… "
  for i in {1..60}; do
    if curl -s "http://127.0.0.1:$VLLM_PORT/get_tensor_parallel_size/" >/dev/null; then echo "READY"; break; fi
    sleep 2
  done
fi

# 2) Lancer le monitoring GPU ------------------------------------------------
echo "📊 Démarrage du monitoring GPU..."
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 10 > gpu_monitoring.log &
MONITOR_PID=$!

# 3) Lancement de l'entraînement RL ----------------------------------------
echo "🚀 Lancement de l'entraînement RLT..."
echo "   Modèle source: $MODEL_PATH"
echo "   (W&B est configuré en mode offline pour éviter les erreurs d'API Key)"
echo "   Logs GPU en temps réel: tail -f gpu_monitoring.log"
echo ""

python3 train.py \
    +run_cfg=teacher_rlt \
    +do_sft=false \
    model_name_or_path="$MODEL_PATH" \
    max_steps=10 \
    wandb_project=rl4lm_teacher_96gb &
TRAIN_PID=$!

# Fonction de nettoyage
cleanup() {
    echo "";
    echo "🛑 Arrêt du training...";
    kill $TRAIN_PID 2>/dev/null || true;
    kill $MONITOR_PID 2>/dev/null || true;
    [ -n "${VLLM_PID:-}" ] && kill $VLLM_PID 2>/dev/null || true;
    echo "📊 Log GPU sauvegardé : gpu_monitoring.log";
}
trap cleanup EXIT INT TERM

echo "📊 Consultez les logs: training_*.log et gpu_monitoring.log"
echo ""
echo "📁 Structure des répertoires de sortie:"
echo "   - results/step-1_SFT_[timestamp]/     # Adaptateurs LoRA du SFT"
echo "   - results/step-1_SFT_fused/           # Modèle SFT fusionné"
echo "   - results/step-2_RLT_[timestamp]/     # Résultats de l'entraînement RL"
echo ""

# Attendre la fin de l'entraînement
echo "⏳ Entraînement en cours... (PID: $TRAIN_PID)"
wait $TRAIN_PID
TRAIN_EXIT_CODE=$?

# Arrêter le monitoring
kill $MONITOR_PID 2>/dev/null || true

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉 Entraînement RLT terminé avec succès !"
    echo "📁 Résultats disponibles dans: results/step-2_RLT_*/"
else
    echo ""
    echo "❌ Entraînement RLT échoué (code de sortie: $TRAIN_EXIT_CODE)"
    echo "📊 Consultez les logs pour plus de détails"
    exit $TRAIN_EXIT_CODE
fi 