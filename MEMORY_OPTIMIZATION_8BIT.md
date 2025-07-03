# 🚀 Optimisation Mémoire avec Quantification 8-bit

## 📊 **Économies Spectaculaires**

La quantification 8-bit (`load_in_8bit=True`) réduit drastiquement l'usage VRAM :

```
AVANT (bfloat16):      APRÈS (8-bit):        ÉCONOMIE:
├── Teacher: 14GB  →   ├── Teacher: 7GB   →   -7GB (-50%)
├── Student: 14GB  →   ├── Student: 7GB   →   -7GB (-50%)  
├── Reference: 14GB →   ├── Reference: 7GB →   -7GB (-50%)
└── TOTAL: 42GB        └── TOTAL: 21GB       -21GB (-50%)

MÉMOIRE TOTALE:
• Sans 8-bit: ~68GB (marge 17GB)
• Avec 8-bit: ~50GB (marge 35GB) 
• GAIN: +18GB de marge supplémentaire!
```

## 🎯 **Configurations Disponibles**

### 1. **Configuration Conservative (Recommandée)**
```bash
python train.py --config-path=cfgs/run_cfg --config-name=teacher_rlt_mono_85gb
```
- **Usage**: ~50GB (35GB libre)
- **Paramètres**: batch_size=64, generations=16, context=6144  
- **Pour**: Premier entraînement, tests de stabilité

### 2. **Configuration Aggressive (Performance Max)**
```bash
python train.py --config-path=cfgs/run_cfg --config-name=teacher_rlt_mono_85gb_aggressive
```
- **Usage**: ~65GB (20GB libre)
- **Paramètres**: batch_size=128, generations=32, context=8192
- **Pour**: Maximiser performance quand le système est stable

## ⚡ **Avantages de la Quantification 8-bit**

### **✅ Bénéfices:**
- **50% réduction** de l'usage mémoire des modèles
- **Inference plus rapide** (moins de transferts mémoire)
- **Compatibilité** excellente avec bitsandbytes
- **Perte de précision** < 1-2% (négligeable)

### **⚠️ Considérations:**
- **Premier chargement** légèrement plus lent (quantification)
- **CPU usage** temporaire durant la quantification
- **Compatible** uniquement avec GPU récents (Ampere+)

## 🔧 **Configuration Technique**

La quantification 8-bit est activée via :

```yaml
model_args:
  load_in_8bit: true      # Quantification 8-bit
  device_map: auto        # Mapping automatique des couches
  torch_dtype: bfloat16   # Dtype pour les calculs
```

## 📈 **Recommandations d'Usage**

1. **Commencer** avec la configuration conservative
2. **Monitorer** l'usage mémoire avec `nvidia-smi`
3. **Passer** à aggressive si stable et besoin de performance
4. **Ajuster** batch_size selon les résultats d'entraînement

## 🎮 **Commandes de Monitoring**

```bash
# Surveiller VRAM en temps réel
watch -n 1 nvidia-smi

# Logger usage mémoire
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 5 > memory_log.csv
``` 