# Quick Start: Two-Stage PQ-VAE Training

## ⚡ Fast Track (Copy-Paste Commands)

### Prerequisites Check
```bash
# 1. Check patch embeddings exist
ls dataset/amazon/2014/processed/beauty_text_patches.npy
ls dataset/amazon/2014/processed/beauty_text_masks.npy

# If missing, run patch processor:
python -m data.patch_processor dataset/amazon/2014 beauty
```

### Stage 1: Item-Level Training

```bash
# Submit job
sbatch job_scripts/train_pqvae_stage1.job

# Monitor progress
tail -f slurm_output/train/pqvae/stage1_beauty_*.out

# Or run interactively (for debugging)
python train_pqvae_stage1.py configs/pq_stage1_beauty.gin
```

**Expected Output**:
- Training will run for 50,000 iterations (~6-8 hours on H100)
- Checkpoints saved every 10,000 iterations
- Final checkpoint: `~/logdir/pqvae/stage1/beauty/{timestamp}/checkpoint_50000.pt`

### Stage 2: Sequence-Level Training

```bash
# 1. Copy checkpoint path from Stage 1
STAGE1_CKPT="$(ls -td ~/logdir/pqvae/stage1/beauty/*/checkpoint_50000.pt | head -1)"
echo "Stage 1 checkpoint: $STAGE1_CKPT"

# 2. Update config (automated)
sed -i "s|train.stage1_checkpoint_path=.*|train.stage1_checkpoint_path=\"$STAGE1_CKPT\"|" configs/pq_stage2_beauty.gin

# 3. Submit job
sbatch job_scripts/train_pqvae_stage2.job

# Monitor
tail -f slurm_output/train/pqvae/stage2_beauty_*.out
```

**Expected Output**:
- Training will run for 30,000 iterations (~4-6 hours)
- Final checkpoint: `~/logdir/pqvae/stage2/beauty/{timestamp}/checkpoint_30000.pt`
- Use this for decoder training!

---

## 📋 Checklist

### Before Starting

- [ ] Patch embeddings preprocessed (`beauty_text_patches.npy` exists)
- [ ] Dataset processed (`data_beauty.pt` exists)  
- [ ] CUDA available (`nvidia-smi` works)
- [ ] WandB API key set (in job scripts)
- [ ] Conda environment activated (`rq-vae`)

### Stage 1 Complete

- [ ] Checkpoint saved: `checkpoint_50000.pt`
- [ ] WandB shows decreasing `train/patch_recon` loss
- [ ] WandB shows `train/p_unique_ids` > 0.8 (good codebook usage)
- [ ] Config updated for Stage 2 with checkpoint path

### Stage 2 Complete

- [ ] Checkpoint saved: `checkpoint_30000.pt`
- [ ] WandB shows decreasing `train/seq_contrastive` loss
- [ ] Ready for decoder training

---

## 🔍 Validation

### Test Stage 1 Model

```python
import torch
from modules.pqvae_updated import PqVae

# Load checkpoint
ckpt = torch.load("path/to/checkpoint_50000.pt")

# Initialize model
model = PqVae(
    num_codebooks=4,
    codebook_size=256,
    patch_token_embed_dim=192,
)

# Load weights
model.load_state_dict(ckpt['model'])
model.eval()

# Test semantic ID extraction
from data.processed import ItemData
dataset = ItemData("dataset/amazon/2014", split="beauty", use_patch_embeddings=True)

batch = dataset[0]
output = model.get_semantic_ids(batch)
print("Semantic IDs shape:", output.sem_ids.shape)  # Should be [1, 4]
print("Semantic IDs:", output.sem_ids)  # Should be 4 integers (0-255 each)
```

### Test Stage 2 Model

```python
from modules.pqvae_twostage import PqVaeTwoStage

# Load Stage 2 checkpoint
model = PqVaeTwoStage(...)
model.load_pretrained("path/to/stage2/checkpoint_30000.pt")

# Check both stages work
model.set_stage(1)  # Item-level
model.set_stage(2)  # Sequence-level
print("✓ Both stages functional")
```

---

## 🐛 Common Errors & Fixes

### Error: "text_patches required"

**Fix**:
```bash
python -m data.patch_processor dataset/amazon/2014 beauty --force
```

### Error: "Stage 1 checkpoint not found"

**Fix**:
```bash
# Find checkpoint
find ~/logdir/pqvae/stage1 -name "checkpoint_*.pt"

# Update config with full path
vim configs/pq_stage2_beauty.gin
```

### Error: CUDA OOM

**Fix** (reduce memory):
```bash
# In config:
train.batch_size=256  # Was 512
train.patch_max_seq_length=64  # Was 77
```

### Error: "No sequences sampled"

**Fix** (increase sampling):
```bash
# In Stage 2 config:
train.sequence_sample_size=50  # Was 10
```

---

## 📊 Expected Training Times

| Stage | Iterations | H100 Time | A100 Time | V100 Time |
|-------|-----------|-----------|-----------|-----------|
| 1     | 50,000    | 6-8 hrs   | 10-12 hrs | 18-24 hrs |
| 2     | 30,000    | 4-6 hrs   | 6-8 hrs   | 12-16 hrs |
| **Total** | **80,000** | **10-14 hrs** | **16-20 hrs** | **30-40 hrs** |

*Assumes batch_size=512, default settings*

---

## 🎯 Key Hyperparameters

### Stage 1 (Item-Level)

```python
patch_recon_weight = 1.0      # MAIN: Don't change!
global_recon_weight = 0.2     # AUXILIARY: Keep low (0.1-0.3)
commitment_weight = 0.25      # Standard VQ-VAE
diversity_weight = 0.01       # Encourage variety
```

**Tuning Tips**:
- If patch reconstruction not improving → increase `patch_recon_weight` (try 1.5)
- If codes collapse (low unique usage) → increase `diversity_weight` (try 0.02)

### Stage 2 (Sequence-Level)

```python
sequence_contrastive_weight = 0.5  # Balance with reconstruction
contrastive_temperature = 0.1      # Sharpness of similarity
num_negatives = 16                 # Negative samples per anchor
```

**Tuning Tips**:
- If seq contrastive dominates → reduce weight (try 0.3)
- If not learning behavioral patterns → increase weight (try 0.7)
- If training unstable → increase temperature (try 0.2)

---

## 📈 Monitoring Metrics

### WandB Dashboard

Stage 1 - Look for:
- ✅ `train/patch_recon`: Steadily decreasing (target: <0.1)
- ✅ `train/global_recon`: Lower than patch (target: <0.05)
- ✅ `train/p_unique_ids`: High usage (target: >0.8)

Stage 2 - Additionally:
- ✅ `train/seq_contrastive`: Decreasing (target: <0.5)
- ✅ Reconstruction losses stay stable (not degrading)

**Red Flags** 🚩:
- Patch recon increasing → learning rate too high
- Unique IDs dropping → codebook collapse, increase diversity
- Seq contrastive stuck → not enough sequence samples

---

## 🚀 Next Steps

After Stage 2 completes:

```bash
# 1. Update decoder config with Stage 2 checkpoint
vim configs/decoder_patches_gin

# Set:
train.pretrained_rqvae_path="~/logdir/pqvae/stage2/beauty/{timestamp}/checkpoint_30000.pt"

# 2. Train decoder
sbatch job_scripts/train_decoder.job

# 3. Evaluate
python test_decoder.py
```

---

## 💡 Pro Tips

1. **Use tmux for long jobs**:
   ```bash
   tmux new -s pqvae
   python train_pqvae_stage1.py configs/pq_stage1_beauty.gin
   # Detach: Ctrl+B, D
   # Reattach: tmux attach -t pqvae
   ```

2. **Compare loss components**:
   ```python
   # In WandB, create custom chart:
   # X-axis: iteration
   # Y-axis: train/patch_recon, train/global_recon (separate lines)
   # Should see patch_recon >> global_recon
   ```

3. **Checkpoint every 5k during development**:
   ```python
   train.save_model_every=5000  # More frequent saves
   ```

4. **Quick test before full run**:
   ```python
   train.iterations=1000  # Test for 1k iterations (~10 min)
   ```

---

Happy training! 🎉