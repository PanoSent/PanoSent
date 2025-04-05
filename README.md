### 1. Code Structure 

```
Sentica/
├── assets/
├── checkpoints/
├── data/
│   └── T-X_pair_data/
│       ├── LLaVA/
│       ├── miniGPT-4/
│       └── VideoChat/
├── sentica/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── imagebind_encoder.py
│   │   ├── flant5_model.py
│   │   └── projection_layer.py
│   ├── utils.py
│   └── datasets/
│       ├── stage1_caption_dataset.py
│       ├── stage2_sextuple_dataset.py
│       └── stage3_entailment_dataset.py
├── scripts/
│   ├── train_stage1.sh
│   ├── train_stage2.sh
│   └── train_stage3.sh
├── train.py
├── requirements.txt
```
