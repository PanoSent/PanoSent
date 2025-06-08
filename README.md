<br />
<p align="center">
  <h1 align="center">PanoSent: A Panoptic Sextuple Extraction Benchmark for Multimodal Conversational Aspect-based Sentiment Analysis</h1>
  <p align="center">
    <a href="https://eurekaleo.github.io/">Meng Luo</strong></a>
    ·
    <a href="https://haofei.vip/">Hao Fei</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=90mnP8MAAAAJ&hl=en">Bobo Li</strong></a>
    ·
    <a href="https://chocowu.github.io/">Shengqiong Wu</strong></a>
    ·
    <a href="https://profiles.auckland.ac.nz/liu-qian">Qian Liu</strong></a>
    ·
    <br/>
    <a href="https://scholar.google.com.sg/citations?user=oS6gRc4AAAAJ&hl=en">Soujanya Poria</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=ilSYpW0AAAAJ&hl=en">Erik Cambria</strong></a>
    ·
    <a href="https://www.comp.nus.edu.sg/~leeml/">Mong-Li Lee</strong></a>
    ·
    <a href="https://www.comp.nus.edu.sg/~whsu/">Wynne Hsu</strong></a>
  </p>
  <p align="center" margin="0 auto">
    <small>National University of Singapore · Wuhan University · The University of Auckland · 
    <br/> Singapore University of Technology and Design · Nanyang Technological University</small>
  </p>




  
  
  <p align="center">
    <a href='https://www.arxiv.org/pdf/2408.09481'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'> </a>
    <a href='https://panosent.github.io/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'> </a>
  </p>
<br />

![avatar](./poster.png)


### Abstract
While existing Aspect-based Sentiment Analysis (ABSA) has received extensive effort and advancement, there are still gaps in defining a more holistic research target seamlessly integrating multimodality, conversation context, fine-granularity, and also covering the changing sentiment dynamics as well as cognitive causal rationales. This paper bridges the gaps by introducing a multimodal conversational ABSA, where two novel subtasks are proposed: Panoptic Sentiment Sextuple Extraction, panoramically recognizing holder, target, aspect, opinion, sentiment, rationale from multi-turn multi-party multimodal dialogue. Sentiment Flipping Analysis, detecting the dynamic sentiment transformation throughout the conversation with the causal reasons. To benchmark the tasks, we construct PanoSent, a dataset annotated both manually and automatically, featuring high quality, large scale, multimodality, multilingualism, multi-scenarios, and covering both implicit & explicit sentiment elements. To effectively address the tasks, we devise a novel Chain-of-Sentiment reasoning framework, together with a novel multimodal large language model (namely Sentica) and a paraphrase-based verification mechanism. Extensive evaluations demonstrate the superiority of our methods over strong baselines, validating the efficacy of all our proposed methods. The work is expected to open up a new era for the ABSA community.

### Sentica
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
```

<span id='Prepare Pre-trained Checkpoint'/>

### 3. Preparing Pre-trained Checkpoints 

- **ImageBind**  
  Download the official `imagebind_huge.pth` checkpoint from [here](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth), and place it at:
  
  ```
  ./imagebind/imagebind_huge.pth
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

…

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
