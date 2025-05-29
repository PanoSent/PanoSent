# Sentica MLLM

We develop a novel MLLM, Sentica, which adopts the FlanT5 (XXL) as the core LLM for semantics understanding and decision-making. For non-text inputs, we use multimodal models to encode signals into LLM-understandable representations. We use ImageBind as the unified encoder for all three non-text modalities due to its strong capabilities, followed by a linear layer that connects ImageBind to the LLM for representation projection.

---

### 1. Code Structure 
```
PanoSent/                     
├── data/
│   ├── T-X_pair_data/                 
│   │   ├── LLaVA/
│   │   ├── miniGPT-4/
│   │   └── VideoChat/
│   ├── PanoSent_train.json            
│   └── PpV_train.json                 
├── PanoSent/
│   ├── model/
│   │   ├── imagebind_encoder.py       
│   │   ├── flant5_model.py          
│   │   ├── projection_layer.py       
│   │   └── lora_utils.py             
│   ├── utils/
│   │   └── imagebind_utils.py        
│   └── datasets/
│       ├── stage1_caption_dataset.py 
│       ├── stage2_sextuple_dataset.py 
│       └── stage3_entailment_dataset.py 
├── scripts/
│   ├── train_stage1.sh               
│   ├── train_stage2.sh               
│   └── …           
├── train.py                           
├── evaluate_subtask1.py              
├── evaluate_subtask2.py               
├── requirements.txt                  
└── README.md
```

### 2. Environment Preparation 

```bash
conda create -n sentica python=3.10
conda activate sentica

git clone https://github.com/PanoSent/PanoSent.git
cd Sentica

pip install -r requirements.txt
```

<span id='Prepare Pre-trained Checkpoint'/>

### 3. Preparing Pre-trained Checkpoints 

- **ImageBind**  
  Download the official `imagebind_huge.pth` checkpoint from [here](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth), and place it at:
  
  ```
  ./checkpoints/imagebind/imagebind_huge.pth
  ```
- **Flan-T5**  
  We use [Flan-T5 XXL](https://huggingface.co/google/flan-t5-xxl) as the LLM backbone.  

<span id='Prepare Dataset'/>

### 4. Preparing Datasets

Sentica consists of three instruction tuning stages. The corresponding datasets are:

#### 4.1 ‘Text+X’ pairs

- `LLaVA`  
- `miniGPT-4`  
- `VideoChat` 

After downloading these datasets, organize them as:

```
./data/T-X_pair_data/
├── LLaVA/
├── miniGPT-4/
└── VideoChat/
```

#### 4.2 PanoSent train set

- `PanoSent_train.json`  

```
./data/PanoSent_train.json
```

#### 4.3 Paraphrase pairs

- `PpV_train.json`  

```
./data/PpV_train.json
```

<span id='Training Sentica'/>

### 5. Training Sentica  

Sentica follows a three-stage training process:

- **Stage 1**: Multimodal Understanding Stage
```bash
bash scripts/train_stage1.sh
```

- **Stage 2**: Sextuple Extraction Understanding
```bash
bash scripts/train_stage2.sh
```

- **Stage 3**: Paraphrase-based Verification
```bash
bash scripts/train_stage3.sh
```

<span id='Evaluation'/>

### 6. Evaluation 

#### Subtask-I: Panoptic Sentiment Sextuple Extraction
```bash
python evaluate_subtask1.py --pred pred.json --gt gold.json
```

#### Subtask-II: Sentiment Flipping Analysis
```bash
python evaluate_subtask2.py --pred pred.json --gt gold.json
```

<span id='Contact'/>

## Contact

If you have any questions or feedback, feel free to open an issue or reach out to us at mluo@u.nus.edu

<span id='Citation'/>

## Citation

```bibtex
@inproceedings{luo2024panosent,
  title={Panosent: A panoptic sextuple extraction benchmark for multimodal conversational aspect-based sentiment analysis},
  author={Luo, Meng and Fei, Hao and Li, Bobo and Wu, Shengqiong and Liu, Qian and Poria, Soujanya and Cambria, Erik and Lee, Mong-Li and Hsu, Wynne},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={7667--7676},
  year={2024}
}
```
