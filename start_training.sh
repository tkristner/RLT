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

# Vérification VRAM (minimum 80GB)
VRAM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM_TOTAL" -lt 80000 ]; then
    echo "❌ Erreur: VRAM insuffisante ($VRAM_TOTAL MB). Minimum requis: 80GB"
    exit 1
fi

echo "✅ VRAM suffisante: ${VRAM_TOTAL} MB"

# Choix de configuration
echo ""
echo "🎯 Choisissez votre niveau de performance :"
echo "1) Conservative (~50GB VRAM) - RECOMMANDÉ"
echo "2) Aggressive (~65GB VRAM)"
read -p "Votre choix (1 ou 2): " config_choice

# Paramètres de base pour 85GB
# Note: Le '+' indique à Hydra d'ajouter la clé si elle n'existe pas.
BASE_PARAMS=(
    "+run_cfg=teacher_rlt"
    "model_name_or_path=results/pre_rl_model"
    "max_steps=200"
    "+model_init_kwargs.load_in_8bit=true"
    "+model_init_kwargs.device_map=auto"
    "+model_args.load_in_8bit=true"
    "use_vllm=true"
    "use_vllm_server=false"
    "offload_untrained_models=true"
    "sync_ref_model=false"
    "wandb_project=rl4lm_teacher_85gb_8bit"
    "gradient_checkpointing=true"
    "fp16=false"
    "bf16=false"
    "+trainer_args.optim=paged_adamw_8bit"
)

# Paramètres spécifiques à la configuration choisie
if [[ "$config_choice" == "2" ]]; then
    echo "⚡ Configuration Aggressive sélectionnée"
    CONFIG_PARAMS=(
        "train_batch_size=32"
        "per_device_train_batch_size=2"
        "generation_aggregation_steps=16"
        "num_generations=24"
        "max_prompt_length=8192"
        "max_completion_length=8192"
        "vllm_gpu_memory_utilization=0.55"
        "output_dir=results/rlt_teacher_85gb_aggressive"
    )
else
    echo "🛡️  Configuration Conservative sélectionnée (défaut)"
    CONFIG_PARAMS=(
        "train_batch_size=32"
        "per_device_train_batch_size=2"
        "gradient_accumulation_steps=16"
        "max_steps=1000"
        "max_prompt_length=2048"
        "max_completion_length=2048"
        "wandb_project=rl4lm_sft_85gb"
        "output_dir=results/pre_rl_model"
    )
fi

echo "✅ Configuration chargée."

# On s'assure que WANDB est en mode offline
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"

# --- ÉTAPE 1: SUPERVISED FINE-TUNING (SFT) ---
echo ""
echo "--- ÉTAPE 1: Lancement du pré-entraînement SFT ---"
echo "Objectif: Créer le modèle de base dans 'results/pre_rl_model'"
echo ""

# Vérification si le modèle SFT existe déjà
if [ -d "results/pre_rl_model" ] && [ -f "results/pre_rl_model/pytorch_model.bin" ]; then
    echo "✅ Le modèle pré-entraîné (SFT) existe déjà dans 'results/pre_rl_model'."
    echo "   Saut de l'étape SFT."
else
    echo "🔧 Le modèle SFT n'existe pas. Lancement de l'entraînement..."
    # On utilise la config SFT existante et on ajoute les optimisations ultimes
    python3 train.py \
        +run_cfg=teacher_sft \
        +do_sft=true \
        +model_args.load_in_8bit=true \
        gradient_checkpointing=true \
        bf16=false \
        per_device_train_batch_size=1 \
        gradient_accumulation_steps=16 \
        max_steps=1000 \
        max_seq_length=2048 \
        packing=false \
        add_text_completions=true \
        wandb_project=rl4lm_sft_85gb \
        output_dir="results/pre_rl_model" \
        +trainer_args.optim=paged_adamw_8bit

    if [ ! -f "results/pre_rl_model/pytorch_model.bin" ]; then
        echo "❌ ERREUR: L'entraînement SFT a échoué. Le fichier modèle n'a pas été créé."
        exit 1
    fi
    
    echo "✅ Étape 1 (SFT) terminée avec succès !"
fi

# --- ÉTAPE 2: REINFORCEMENT LEARNING (RL) ---
echo ""
echo "--- ÉTAPE 2: Lancement de l'entraînement par renforcement (RL) ---"

# Lancement de l'entraînement RL
echo "🚀 Lancement de l'entraînement RLT..."
echo "   (W&B est configuré en mode offline pour éviter les erreurs d'API Key)"
echo "   Logs GPU en temps réel: tail -f gpu_monitoring.log"
echo ""

python3 train.py \
    "${BASE_PARAMS[@]}" \
    "${CONFIG_PARAMS[@]}" \
    2>&1 | tee rl_training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Entraînement RL terminé !"

# Monitoring en arrière-plan
echo "📊 Démarrage du monitoring GPU..."
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 10 > gpu_monitoring.log &
MONITOR_PID=$!

# Fonction de nettoyage
cleanup() {
    echo ""
    echo "🛑 Arrêt du training..."
    kill $MONITOR_PID 2>/dev/null || true
    echo "📊 Log GPU sauvegardé : gpu_monitoring.log"
}
trap cleanup EXIT

echo "📊 Consultez les logs: training_*.log et gpu_monitoring.log" 