#!/bin/bash
# Script de vérification de l'organisation des répertoires de sortie RLT

echo "🔍 Vérification de l'organisation des répertoires de sortie RLT"
echo "=================================================================="

# Fonction pour afficher la taille d'un répertoire
get_dir_size() {
    if [ -d "$1" ]; then
        du -sh "$1" 2>/dev/null | cut -f1
    else
        echo "N/A"
    fi
}

# Fonction pour vérifier si un modèle est présent
check_model_files() {
    local dir="$1"
    if [ -d "$dir" ]; then
        local has_pytorch=$([ -f "$dir/pytorch_model.bin" ] && echo "✅" || echo "❌")
        local has_safetensors=$([ -f "$dir/model.safetensors" ] && echo "✅" || echo "❌")
        local has_sharded_safetensors=$(ls "$dir"/model-*-of-*.safetensors 2>/dev/null | head -1 >/dev/null && echo "✅" || echo "❌")
        local has_adapter=$([ -f "$dir/adapter_model.safetensors" ] && echo "✅" || echo "❌")
        local has_config=$([ -f "$dir/config.json" ] && echo "✅" || echo "❌")
        local has_tokenizer=$([ -f "$dir/tokenizer.json" ] && echo "✅" || echo "❌")
        
        echo "    📄 pytorch_model.bin: $has_pytorch"
        echo "    📄 model.safetensors: $has_safetensors"
        echo "    📄 model-*-of-*.safetensors: $has_sharded_safetensors"
        echo "    📄 adapter_model.safetensors: $has_adapter"
        echo "    📄 config.json: $has_config"
        echo "    📄 tokenizer.json: $has_tokenizer"
    else
        echo "    ❌ Répertoire inexistant"
    fi
}

echo ""
echo "📁 STRUCTURE ACTUELLE DES RÉPERTOIRES:"
echo ""

# Vérifier le répertoire results principal
if [ -d "results" ]; then
    echo "📂 results/ ($(get_dir_size results))"
    
    # Lister tous les répertoires step-1_SFT
    echo ""
    echo "🔸 ÉTAPE 1 - SFT (Supervised Fine-Tuning):"
    
    sft_found=false
    for dir in results/step-1_SFT_*/; do
        if [ -d "$dir" ] && [[ ! "$dir" =~ step-1_SFT_fused ]]; then
            sft_found=true
            echo "  📂 $(basename "$dir") ($(get_dir_size "$dir"))"
            check_model_files "$dir"
            echo ""
        fi
    done
    
    # Vérifier le modèle fusionné
    if [ -d "results/step-1_SFT_fused" ]; then
        echo "  📂 step-1_SFT_fused ($(get_dir_size "results/step-1_SFT_fused")) - MODÈLE FUSIONNÉ"
        check_model_files "results/step-1_SFT_fused"
        echo ""
    else
        echo "  ❌ step-1_SFT_fused - Modèle fusionné non trouvé"
        echo ""
    fi
    
    if [ "$sft_found" = false ]; then
        echo "  ❌ Aucun répertoire step-1_SFT trouvé"
        echo ""
    fi
    
    # Lister tous les répertoires step-2_RLT
    echo "🔸 ÉTAPE 2 - RLT (Reinforcement Learning Teachers):"
    
    rlt_found=false
    for dir in results/step-2_RLT_*/; do
        if [ -d "$dir" ]; then
            rlt_found=true
            echo "  📂 $(basename "$dir") ($(get_dir_size "$dir"))"
            check_model_files "$dir"
            echo ""
        fi
    done
    
    if [ "$rlt_found" = false ]; then
        echo "  ❌ Aucun répertoire step-2_RLT trouvé"
        echo ""
    fi
    
    # Vérifier l'ancienne structure
    echo "🔸 ANCIENNE STRUCTURE (à nettoyer):"
    
    old_found=false
    for dir in results/rl4lm*/ results/pre_rl_model/ results/rlt_teacher/; do
        if [ -d "$dir" ]; then
            old_found=true
            echo "  📂 $(basename "$dir") ($(get_dir_size "$dir")) - ANCIENNE STRUCTURE"
        fi
    done
    
    if [ "$old_found" = false ]; then
        echo "  ✅ Aucune ancienne structure trouvée"
    fi
    
else
    echo "❌ Répertoire 'results' non trouvé"
fi

echo ""
echo "🔧 RECOMMANDATIONS:"
echo ""

# Vérifier si on peut démarrer l'entraînement
if [ -d "results/step-1_SFT_fused" ] && ([ -f "results/step-1_SFT_fused/pytorch_model.bin" ] || [ -f "results/step-1_SFT_fused/model.safetensors" ] || ls results/step-1_SFT_fused/model-*-of-*.safetensors 1> /dev/null 2>&1); then
    echo "✅ Modèle SFT fusionné prêt pour l'étape RL"
    echo "   Commande: ./start_training.sh"
elif ls results/step-1_SFT_*/adapter_model.safetensors 1> /dev/null 2>&1; then
    latest_sft=$(ls -td results/step-1_SFT_*/ | head -1)
    echo "🔄 Adaptateurs LoRA SFT trouvés, fusion nécessaire"
    echo "   Répertoire: $latest_sft"
    echo "   Commande: ./start_training.sh (fusionnera automatiquement)"
else
    echo "🚀 Aucun modèle SFT trouvé, démarrer l'entraînement complet"
    echo "   Commande: ./start_training.sh"
fi

echo ""
echo "🧹 NETTOYAGE (optionnel):"
echo "   rm -rf results/rl4lm*"
echo "   rm -rf results/pre_rl_model"
echo "   rm -rf results/rlt_teacher"
echo "" 