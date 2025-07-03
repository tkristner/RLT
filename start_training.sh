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
        use_peft=true \
        +model_args.load_in_8bit=true \
        gradient_checkpointing=true \
        bf16=false \
        max_steps=200 \
        max_seq_length=2048 \
        packing=false \
        add_text_completions=true \
        wandb_project=rl4lm_sft_85gb \
        +trainer_args.optim=paged_adamw_8bit

    if [ ! -f "SFT_model_pre_RL/adapter_model.safetensors" ]; then
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