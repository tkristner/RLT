#!/usr/bin/env python3
"""
Test script pour vérifier la quantification FP8 avec vLLM
"""
import sys
import torch
from pathlib import Path

def test_fp8_quantization():
    """Test la quantification FP8 avec vLLM"""
    
    print("🧪 Test de la quantification FP8 avec vLLM")
    print("=" * 50)
    
    # Vérifier la compatibilité GPU
    if not torch.cuda.is_available():
        print("❌ CUDA non disponible")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    print(f"🔧 GPU: {gpu_name}")
    print(f"🔧 Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
    
    # Vérifier si FP8 est supporté (compute capability >= 8.9)
    if compute_capability[0] < 8 or (compute_capability[0] == 8 and compute_capability[1] < 9):
        print("⚠️  Attention: FP8 nécessite compute capability >= 8.9")
        print("   Votre GPU utilisera probablement W8A16 (poids FP8, activations FP16)")
    else:
        print("✅ GPU compatible FP8 W8A8")
    
    # Test d'importation vLLM
    try:
        from vllm import LLM
        print("✅ vLLM importé avec succès")
    except ImportError as e:
        print(f"❌ Erreur d'importation vLLM: {e}")
        return False
    
    # Test de modèle simple avec quantification FP8
    try:
        print("\n🚀 Test de quantification FP8...")
        
        # Utiliser un modèle très petit pour le test
        test_model = "microsoft/DialoGPT-small"  # ~117MB
        
        llm = LLM(
            model=test_model,
            quantization="fp8",
            gpu_memory_utilization=0.3,
            max_model_len=512,
            dtype="auto"
        )
        
        print("✅ Modèle chargé avec quantification FP8")
        
        # Test de génération simple
        outputs = llm.generate(["Hello"], sampling_params={"max_tokens": 5, "temperature": 0.8})
        print(f"✅ Génération test réussie: {outputs[0].outputs[0].text}")
        
        # Nettoyage
        del llm
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test FP8: {e}")
        return False

def test_model_loading():
    """Test le chargement du modèle principal"""
    
    print("\n🔍 Vérification du modèle principal...")
    
    # Vérifier si le modèle SFT existe
    sft_model_path = Path("results/step-1_SFT_fused")
    if sft_model_path.exists():
        print(f"✅ Modèle SFT trouvé: {sft_model_path}")
        
        # Lister les fichiers du modèle
        model_files = list(sft_model_path.glob("*.bin")) + list(sft_model_path.glob("*.safetensors"))
        if model_files:
            total_size = sum(f.stat().st_size for f in model_files) / (1024**3)
            print(f"📊 Taille du modèle: {total_size:.1f} GB")
        
        return True
    else:
        print(f"❌ Modèle SFT non trouvé: {sft_model_path}")
        return False

def main():
    """Fonction principale"""
    
    print("🧪 Tests de Quantification FP8 pour RLT")
    print("=" * 60)
    
    # Test 1: Compatibilité FP8
    fp8_ok = test_fp8_quantization()
    
    # Test 2: Modèle principal
    model_ok = test_model_loading()
    
    print("\n📋 Résumé des tests:")
    print(f"   FP8 Quantization: {'✅' if fp8_ok else '❌'}")
    print(f"   Modèle principal: {'✅' if model_ok else '❌'}")
    
    if fp8_ok and model_ok:
        print("\n🎉 Tous les tests réussis ! Vous pouvez utiliser FP8.")
        print("\n💡 Configuration recommandée:")
        print("   - vllm_quantization: fp8")
        print("   - vllm_gpu_memory_utilization: 0.4")
        print("   - vllm_max_model_len: 8192")
    else:
        print("\n⚠️  Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
    
    return fp8_ok and model_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 