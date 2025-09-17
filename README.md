# 🐅 TIGRESS: Transformer Index for Generative Recommenders with Enhanced Semantic Signals


## Setup and Installation

### 1. Create Environment

```bash
conda create -n rq-vae python=3.9
pip install -r requirements.txt
conda activate rq-vae
```

or using SLURM job

```bash
sbatch run job_scripts/install_enviroment.job
```

### 2. Weights & Biases (Optional)
Get your key from: [link](https://wandb.ai/authorize)
```bash
wandb login <API Key>
```

### 3. Dataset Downloading & Processing

```bash
# change directory
cd data

# amazon 2014 (auto-downloads (P5 processed data), process splits (train:dev:test), and save)
python amazon.py --root ../dataset/amazon/2014 --split beauty

# amazon 2023 (auto-downloads (raw data), pre-process (P5), and save)
python p5.py \
    --dataset_dir "../dataset/amazon/2023/raw" \
    --dataset_split "beauty" \
    --data_type "Amazon"

# process splits (train:dev:test) and save
python amazon.py --root ../dataset/amazon/2023 --split beauty

# download product images (2014 and 2023)
python download_product_images.py \
    --dataset_dir "../dataset/amazon/2023/raw" \
    --dataset_split "beauty"
```

### 4. Run Training and Testing

Configs `(configs/*.gin)` can be modified to adapt to different datasets, categories, and other parameters.

For example,
```python
train.dataset_folder="dataset/amazon/2014"
train.dataset_split="beauty"
train.log_dir="logdir/rqvae/amazon/2014"
train.wandb_logging=False
train.use_image_features=False # for CLIP-based semantic ids
train.feature_combination_mode="" # "sum", "concat" or "cross-attn" if use_image_features
train.run_prefix="" # for wandb
train.pretrained_decoder_path=None
train.pretrained_rqvae_path=None
train.enable_image_cross_attn=False # use cross attention (text and image)
```

To start the training,
```bash
# training RQ-VAE
python train_rqvae.py --config configs/rqvae_amazon.gin

# training Encoder-Decoder-Model
python train_decoder.py --config configs/decoder_amazon.gin
```
or using SLURM job

```bash
sbatch run job_scripts/train_rqvae.job
sbatch run job_scripts/train_decoder.job
sbatch run job_scripts/test_decoder.job
```


## Weights & Biases Results Dashboard
- RQ-VAE Training Report: [link](https://wandb.ai/RecSys-UvA/rq-vae-training/reports/RQ-VAE-Training-Report--VmlldzoxMzM2NjQ5MA?accessToken=ycktjvkde9hfnfv7gz7zlkcjqtsl4pr2c1x3sy65megn49yebpi9nu3vwjzwcpt3)
- Decoder Training and Testing Report: [link](https://api.wandb.ai/links/RecSys-UvA/ofxtg8fq)
- Decoder Testing Only Report: [link](https://wandb.ai/RecSys-UvA/gen-ir-decoder-testing/reports/Decoder-Testing-Report--VmlldzoxMzM2NjQ1Mw?accessToken=r3cvpvonokw0kbslg5bt10kht3yf8chxhk7gjkpi2dt56wwaw9mstsn9qgwo2o96)

## High-Level Directory Structure

```bash
├── assets/                # Visuals used in paper/README (eg., plots, diagrams)
├── configs/               # Configuration files for RQ-VAE and decoder training
├── data/                  # Data loaders, preprocessing scripts, schema
├── evaluate/              # Evaluation metrics and logic
├── init/                  # Initialization strategies (e.g., KMeans)
├── job_scripts/           # SLURM job scripts for training and evaluation
├── metrics/               # Stored evaluation results (e.g., Recall, Fairness)
├── modules/               # Core model components (encoder, quantizer, transformer, etc.)
├── notebooks/             # Jupyter notebooks for analysis and preprocessing
├── ops/                   # Triton for jagged transformer integration
├── train_rqvae.py         # Script to train the RQ-VAE model
├── train_decoder.py       # Script to train the Transformer decoder
├── test_decoder.py        # Script to evaluate the decoder
├── requirements.txt       # Python dependencies
├── README.md              # Project overview and usage
├── REPRO.md               # Reproducibility checklist
└── RQ-VAE-README.md       # Notes specific to RQ-VAE implementation
```



## Datasets


- [Amazon 2014 [2]](https://jmcauley.ucsd.edu/data/amazon/index_2014.html)
  - Pre-processing: We used [P5 Pre-procesing](https://github.com/jeykigung/P5/blob/main/preprocess/data_preprocess_amazon.ipynb) pipeline to pre-process the data and keep users interactions having equal to and more than 5 reviews.
  - Categories considered: Beauty, Sports and Outdoors, Toys and Games
  - Attributes for user fairness: User Interaction History (Title, Brand, Categories, Price)
  - Attributes for item fairness: Title, Brand, Categories, Price
  - Other attributes: Product Images

- [Amazon 2023 [3]](https://amazon-reviews-2023.github.io/)
  - Pre-processing: Similar to the 2014 edition, we used [P5 Pre-procesing](https://github.com/jeykigung/P5/blob/main/preprocess/data_preprocess_amazon.ipynb) pipeline to pre-process the data and keep users interactions having equal to and more than 5 reviews.
  - Categories considered: All Beauty, Sports and Outdoors, Toys and Games, Video Games, Software
  - Attributes for user fairness: User Interaction History (Title, Brand, Categories, Rating, Price)
  - Attributes for item fairness: Title, Brand, Categories, Rating, Price
  - Other attributes: Product Images

![Amazon Dataset (2014 & 2023)](assets/datasets.png)
*User interaction statistics for the Amazon 2014 [2] and Amazon 2023 [3] datasets.*


## Baselines

We compare against three sequential recommendation models for reproducibility and benchmarking. For TIGRESS, we focus on comparison with TIGER. All models are implemented using **PyTorch**, and we follow original implementations when available to ensure reproducibility.

- **[SASRec [4]](https://github.com/pmixer/SASRec.pytorch)**: A transformer-based sequential recommender that uses self-attention to model user-item interaction sequences. Serves as a strong baseline for modeling user preferences via item embeddings.

- **[S³-Rec [5]](https://github.com/RUCAIBox/CIKM2020-S3Rec)**: Builds upon SASRec by introducing self-supervised learning objectives using mutual information maximization, enhancing robustness and performance in sparse data settings.

- **[TIGER [1]](https://github.com/EdoardoBotta/RQ-VAE-Recommender):** A generative retrieval framework that learns to generate relevant item identifiers directly from user context, integrating a language modeling paradigm into recommendation tasks. Used as the main baseline to evaluate our proposed TIGRESS model.


## References

[1] Rajput, S., Mehta, N., Singh, A., Keshavan, R.H., Vu, T., Heldt, L., Hong, L., Tay, Y., Tran, V.Q., Samost, J., Kula, M., Chi, E.H., Sathiamoorthy, M.: Recommender systems with generative retrieval (2023), https://arxiv.org/abs/2305.05065

[2] McAuley, J., Targett, C., Shi, Q., van den Hengel, A.: Image-based recommendations on styles and substitutes (2015), https://arxiv.org/abs/1506.04757  

[3] Hou, Y., Li, J., He, Z., Yan, A., Chen, X., McAuley, J.: Bridging language and items for retrieval and recommendation. arXiv preprint arXiv:2403.03952 (2024)  

[4] Kang, W.C., McAuley, J.: Self-attentive sequential recommendation. In: 2018 IEEE International Conference on Data Mining (ICDM). pp. 197–206. IEEE (2018), https://arxiv.org/abs/1808.09781  

[5] Kang, W.C., McAuley, J.: Self-attentive sequential recommendation. In: 2018 IEEE International Conference on Data Mining (ICDM). pp. 197–206. IEEE (2018), https://arxiv.org/abs/1808.09781  

[6] Liu, H., Wei, Y., Song, X., Guan, W., Li, Y.F., Nie, L.: Mmgrec: Multimodal generative recommendation with transformer model (2024), https://arxiv.org/abs/ 2404.16555  

[7] Zhai, J., Mai, Z.F., Wang, C.D., Yang, F., Zheng, X., Li, H., Tian, Y.: Multimodal quantitative language for generative recommendation (2025), https://arxiv.org/ abs/2504.05314
