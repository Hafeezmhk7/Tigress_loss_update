import gin
import os
import random
import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.amazon import AmazonReviews
from data.ml1m import RawMovieLens1M
from data.ml32m import RawMovieLens32M
from data.schemas import SeqBatch
from enum import Enum
from torch import Tensor
from torch.utils.data import Dataset
from typing import Optional, Tuple
import logging
from PIL import Image
from torch import nn
import json
import clip
from tqdm import tqdm
import numpy as np

# NEW: Import PatchEmbeddingProcessor
from data.patch_processor import PatchEmbeddingProcessor

# fetch logger
logger = logging.getLogger("recsys_logger")

PROCESSED_MOVIE_LENS_SUFFIX = "/processed/data.pt"


@gin.constants_from_enum
class RecDataset(Enum):
    AMAZON = 1
    ML_1M = 2
    ML_32M = 3


DATASET_NAME_TO_RAW_DATASET = {
    RecDataset.AMAZON: AmazonReviews,
    RecDataset.ML_1M: RawMovieLens1M,
    RecDataset.ML_32M: RawMovieLens32M,
}


DATASET_NAME_TO_MAX_SEQ_LEN = {
    RecDataset.AMAZON: 20,
    RecDataset.ML_1M: 200,
    RecDataset.ML_32M: 200,
}


class CLIPImageEncoder(nn.Module):
    def __init__(self, model_name: str = "ViT-L/14", device: str = "cpu"):
        super().__init__()
        self.model_name = model_name
        self.clip_model, self.preprocess = clip.load(model_name, device=device)
        
        # Freeze CLIP parameters if you don't want to fine-tune
        for param in self.clip_model.parameters():
            param.requires_grad = False  # ← This freezes all parameters
    
    def forward(self, image_input: Tensor) -> Tensor:
        with torch.no_grad():  # ← This also prevents gradients
            image_feat = self.clip_model.encode_image(image_input)
            image_feat = image_feat / image_feat.norm(dim=1, keepdim=True)
        return image_feat


class ItemData(Dataset):
    def __init__(
        self,
        root: str,
        *args,
        force_process: bool = True,
        dataset: RecDataset = RecDataset.ML_1M,
        train_test_split: str = "all",
        use_image_features: bool = False,
        feature_combination_mode: str = "sum",
        device: str = "cpu",
        use_patch_embeddings: bool = False,
        patch_model_name: str = "sentence-transformers/sentence-t5-xl",
        patch_max_seq_length: int = 77,
        **kwargs
    ) -> None:

        self.use_image_features = use_image_features
        self.use_patch_embeddings = use_patch_embeddings
        self.root = root
        self.device = device
        self.feature_combination_mode = feature_combination_mode
        
        raw_dataset_class = DATASET_NAME_TO_RAW_DATASET[dataset]
        max_seq_len = DATASET_NAME_TO_MAX_SEQ_LEN[dataset]
        raw_data = raw_dataset_class(root=self.root, *args, **kwargs)
        processed_data_path = raw_data.processed_paths[0]
        
        if not os.path.exists(processed_data_path) or force_process:
            raw_data.process(max_seq_len=max_seq_len)

        if train_test_split == "train":
            filt = raw_data.data["item"]["is_train"]
        elif train_test_split == "eval":
            filt = ~raw_data.data["item"]["is_train"]
        elif train_test_split == "all":
            filt = torch.ones_like(raw_data.data["item"]["x"][:, 0], dtype=bool)

        # Load item data FIRST
        self.item_data, self.item_text, self.item_brand_id = (
            raw_data.data["item"]["x"][filt],
            raw_data.data["item"]["text"][filt],
            raw_data.data["item"]["brand_id"][filt],
        )
        
        # Set dataset_split SECOND
        self.dataset_split = kwargs.get("split")
        logger.info(f"For `{self.dataset_split}` using the datapath: {processed_data_path}")

        # NOW process patches THIRD (uses self.dataset_split and self.item_text)
        if self.use_patch_embeddings:
            self.patch_processor = PatchEmbeddingProcessor(
                processed_dir=os.path.join(self.root, "processed"),
                dataset_split=self.dataset_split,
                model_name=patch_model_name,
                max_seq_length=patch_max_seq_length,
            )
            
            if self.patch_processor.is_processed and not force_process:
                logger.info(f"Patch embeddings already cached for {self.dataset_split}")
            else:
                logger.info(f"Processing patch embeddings for {self.dataset_split}...")
                item_titles = self.item_text
                logger.info(f"✓ Using {len(item_titles)} item texts from loaded data")
                self.patch_processor.process(item_titles, force_reprocess=force_process)
                logger.info(f"✅ Patch embeddings cached")
        else:
            self.patch_processor = None

        # Image features (existing code)
        if self.use_image_features:
            with open(os.path.join(self.root, "raw", self.dataset_split, "datamaps.json"), "r") as f:
                self.data_maps = json.load(f)

            features_path = os.path.join(
                self.root, "processed", f"{self.dataset_split}_{train_test_split}_item_img_feats.pt"
            )
            
            if os.path.exists(features_path) and not force_process:
                logger.info(f"Loading precomputed image features from {features_path}")
                self.image_features = torch.load(features_path, map_location="cpu")
                logger.info(f"Loaded image features {self.image_features.shape}")
            else:
                self.image_features = self._precompute_image_features()
                os.makedirs(os.path.dirname(features_path), exist_ok=True)
                torch.save(self.image_features, features_path)
# =============================================================================
# HOW TO APPLY THIS FIX:
# =============================================================================
# 1. Open data/processed.py
# 2. Find line ~168: meta_df = pd.read_json(metadata_path, lines=True)
# 3. Replace that SINGLE line with the 7 lines in "FIX STARTS/ENDS HERE"
# =============================================================================

    def __len__(self):
        return self.item_data.shape[0]

    def _precompute_image_features(self):
        logger.info(f"Pre-computing image features for `{self.__class__.__name__}`")
        
        # create CLIP model temporarily for feature extraction
        img_model = CLIPImageEncoder(device=self.device)
        
        image_features = []
        batch_size = 256
        
        for i in tqdm(range(0, len(self), batch_size)):
            batch_end = min(i + batch_size, len(self))
            batch_images = []
            
            for idx in range(i, batch_end):
                img_id = str(idx + 1)
                img_filename = self.data_maps["id2item"][img_id] + ".jpg"
                img_path = os.path.join(self.root, "raw", self.dataset_split, "product_images", img_filename)
        
                try:
                    image = Image.open(img_path).convert("RGB")
                    image_tensor = img_model.preprocess(image)
                except Exception:
                    image_tensor = torch.zeros((3, 224, 224))
                batch_images.append(image_tensor)
            
            # process batch
            batch_images = torch.stack(batch_images).to(self.device)
            with torch.no_grad():
                batch_features = img_model.clip_model.encode_image(batch_images)
                batch_features = batch_features / batch_features.norm(dim=1, keepdim=True)
            
            image_features.append(batch_features.cpu())  # move to CPU to save GPU memory
        
        # concatenate all features
        all_features = torch.cat(image_features, dim=0)
        
        # clean up the temporary model
        del img_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return all_features

    # NEW: Helper method to get patch embeddings
    def get_patch_embeddings(
        self,
        indices: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get patch embeddings for given indices.
        
        Args:
            indices: Item indices
            
        Returns:
            text_patches: [batch_size, max_seq_len, hidden_dim] or None
            text_masks: [batch_size, max_seq_len] or None
        """
        if not self.use_patch_embeddings or self.patch_processor is None:
            return None, None
        
        return self.patch_processor.get_patch_embeddings(indices, device=self.device)

        # FIX: ItemData.__getitem__ should pass dataset index, not item_ids

        # In data/processed.py, find the __getitem__ method in ItemData class
        # and modify the patch embeddings call:

    # FIX: Handle both single index and batch of indices

    # In data/processed.py, ItemData.__getitem__ method:

        # SIMPLE FIX: Convert idx to tensor before using for patch embeddings

    # # In data/processed.py, around line 255, change:

    # # ❌ BEFORE (BROKEN):
    # text_patches, text_masks = self.get_patch_embeddings(idx)  # idx might be a list!

    # # ✅ AFTER (FIXED):
    # idx_tensor = torch.tensor(idx) if isinstance(idx, (list, int)) else idx
    # text_patches, text_masks = self.get_patch_embeddings(idx_tensor)


    # FULL CONTEXT (around line 245-260):

    def __getitem__(self, idx):
        item_ids = (
            torch.tensor(idx).unsqueeze(0) if not isinstance(idx, torch.Tensor) else idx
        )
        x = self.item_data[idx, :768]
        x_image = None
        x_brand_id = torch.tensor([self.item_brand_id[idx]])  # ✅ FIX: Ensure it's always [1]
        
        # ✅ FIX: Get patch embeddings for single item
        if hasattr(self, 'patch_processor') and self.patch_processor is not None:
            idx_tensor = torch.tensor([idx]) if isinstance(idx, int) else torch.tensor(idx)
            text_patches, text_masks = self.get_patch_embeddings(idx_tensor)
            # ✅ CRITICAL: Squeeze batch dimension for single items
            if text_patches is not None and text_patches.dim() > 2:
                text_patches = text_patches.squeeze(0)  # [1, 77, 1024] -> [77, 1024]
            if text_masks is not None and text_masks.dim() > 1:
                text_masks = text_masks.squeeze(0)  # [1, 77] -> [77]
        else:
            text_patches = None
            text_masks = None
        
        return SeqBatch(
            user_ids=-1 * torch.ones_like(item_ids.squeeze(0)),
            ids=item_ids,
            ids_fut=-1 * torch.ones_like(item_ids.squeeze(0)),
            x=x,
            x_image=x_image,
            x_brand_id=x_brand_id,
            x_fut=-1 * torch.ones_like(item_ids.squeeze(0)),
            x_fut_brand_id=-1 * torch.ones_like(item_ids.squeeze(0)),
            seq_mask=torch.ones_like(item_ids, dtype=bool),
            text_patches=text_patches,
            text_masks=text_masks,
        )

# WHY THIS WORKS:
# - idx can be: int, list, or Tensor
# - get_patch_embeddings() expects: Tensor
# - So we convert: list/int → Tensor before calling
# - If already Tensor → keep as is

# EXPLANATION:
# 
# The dataloader can call __getitem__ with different types:
# 
# 1. Single item:
#    dataset[42]  →  idx = 42 (int)
# 
# 2. Batch of items (with BatchSampler):
#    dataset[[0, 1, 2, ..., 63]]  →  idx = [0, 1, 2, ..., 63] (list)
# 
# We need to handle both cases by converting to tensor first!

# EXPLANATION:
# 
# Dataset structure:
#   Position (idx):  0,    1,    2,    ..., 11499
#   Item IDs:       45, 11712, 8234,  ...,  5678
#                   ↑     ↑     ↑             ↑
#                   These are actual product IDs (can be any number!)
# 
# Patch embeddings:
#   Index:  0,    1,    2,    ..., 11499
#   Data:  [emb] [emb] [emb]  ...  [emb]
#          ↑     ↑     ↑             ↑
#          These correspond to dataset positions, not item IDs!
#
# So we must use idx (0-11499), not item_ids (which could be 11712)!


class SeqData(Dataset):
    def __init__(
        self,
        root: str,
        *args,
        subsample: bool = False,
        force_process: bool = False,
        dataset: RecDataset = RecDataset.ML_1M,
        data_split: str = "train",
        use_image_features: bool = False,
        feature_combination_mode: str = "sum",
        device: str = "cpu",
        **kwargs
    ) -> None:

        assert (not subsample) or (data_split=="train"), "Can only subsample on training split."

        raw_dataset_class = DATASET_NAME_TO_RAW_DATASET[dataset]
        max_seq_len = DATASET_NAME_TO_MAX_SEQ_LEN[dataset]

        raw_data = raw_dataset_class(root=root, *args, **kwargs)

        processed_data_path = raw_data.processed_paths[0]
        if not os.path.exists(processed_data_path) or force_process:
            raw_data.process(max_seq_len=max_seq_len)

        self.data_split = data_split
        self.subsample = subsample
        self.sequence_data = raw_data.data[("user", "rated", "item")]["history"][self.data_split]

        if not self.subsample:
            self.sequence_data["itemId"] = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(l[-max_seq_len:]) for l in self.sequence_data["itemId"]],
                batch_first=True,
                padding_value=-1,
            )

        self._max_seq_len = max_seq_len
        self.item_data = raw_data.data["item"]["x"]
        self.item_brand_id = raw_data.data["item"]["brand_id"]
        self.use_image_features = use_image_features
        self.device = device
        self.feature_combination_mode = feature_combination_mode
        self.root = root
        self.dataset_split = kwargs.get("split")
        
        if self.data_split in ["eval", "test"] and "2023" in self.root:
            # keep N%/25k users for faster eval
            if self.dataset_split in ["sports", "toys", "software", "games", "appliances"]:
                split_fraction = 25_000 / len(self.sequence_data["userId"])
                logger.info(f"Splitting the `{self.data_split}` SeqData into `{int(split_fraction*100)}%` to save compute!")
                self.filter_by_user_fraction(split_fraction)
        
        if self.use_image_features:
            with open(os.path.join(self.root, "raw", self.dataset_split, "datamaps.json"), "r") as f:
                self.data_maps = json.load(f)
                
            # image features path
            features_path = os.path.join(
                self.root, "processed", f"{self.dataset_split}_{self.data_split}_seq_img_feats.pt"
            )
            # load pre-computed image features
            if os.path.exists(features_path) and not force_process:
                logger.info(f"Loading precomputed image features from {features_path}")
                self.image_features = torch.load(features_path, map_location="cpu")
                logger.info(f"Loaded image features {self.image_features.shape}")
            else:
                # pre-compute all image features
                self.image_features = self._precompute_image_features()
                os.makedirs(os.path.dirname(features_path), exist_ok=True)
                torch.save(self.image_features, features_path)

    @property
    def max_seq_len(self):
        return self._max_seq_len

    def __len__(self):
        return self.sequence_data["userId"].shape[0]
    
    def filter_by_user_fraction(self, fraction: float = 0.05):
        # get unique user IDs
        all_users = self.sequence_data["userId"].unique()        
        
        # sample fraction of users using numpy
        if hasattr(all_users, 'to_numpy'):
            # if it's a Polars Series
            users_array = all_users.to_numpy()
        else:
            # if it's already a numpy array or tensor
            users_array = np.array(all_users)
        
        n_sample = int(len(users_array) * fraction)
        sampled_indices = np.random.choice(len(users_array), size=n_sample, replace=False)
        sampled_users = users_array[sampled_indices]
        # filter rows to keep only those users
        user_ids = self.sequence_data["userId"]
        sampled_users_tensor = torch.tensor(sampled_users, device=user_ids.device)
        user_mask = torch.isin(user_ids, sampled_users_tensor).view(-1)
        for key in self.sequence_data:
            self.sequence_data[key] = self.sequence_data[key][user_mask]

    def _precompute_image_features(self):
        logger.info(f"Pre-computing image features for `{self.__class__.__name__}`")
        
        # create CLIP model temporarily for feature extraction
        img_model = CLIPImageEncoder(device=self.device)
        
        image_features = []
        batch_size = 256
        id2item_keys = list(self.data_maps["id2item"].keys())
        
        for i in tqdm(range(0, self.item_data.shape[0], batch_size)):
            batch_end = min(i + batch_size, len(id2item_keys))
            batch_images = []
            
            for idx in range(i, batch_end):
                img_id = id2item_keys[idx]
                img_filename = self.data_maps["id2item"][img_id] + ".jpg"
                img_path = os.path.join(self.root, "raw", self.dataset_split, "product_images", img_filename)
        
                try:
                    image = Image.open(img_path).convert("RGB")
                    image_tensor = img_model.preprocess(image)
                except Exception:
                    image_tensor = torch.zeros((3, 224, 224))
                batch_images.append(image_tensor)
            
            # process batch
            batch_images = torch.stack(batch_images).to(self.device)
            with torch.no_grad():
                batch_features = img_model.clip_model.encode_image(batch_images)
                batch_features = batch_features / batch_features.norm(dim=1, keepdim=True)
            
            image_features.append(batch_features.cpu())  # move to CPU to save GPU memory
        
        # concatenate all features
        all_features = torch.cat(image_features, dim=0)
        
        # clean up the temporary model
        del img_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return all_features
    
    def __getitem__(self, idx):
        user_ids = self.sequence_data["userId"][idx]

        if self.subsample:
            seq = (
                self.sequence_data["itemId"][idx]
                + self.sequence_data["itemId_fut"][idx].tolist()
            )
            start_idx = random.randint(0, max(0, len(seq) - 3))
            end_idx = random.randint(start_idx + 3, start_idx + self.max_seq_len + 1)
            sample = seq[start_idx:end_idx]

            item_ids = torch.tensor(
                sample[:-1] + [-1] * (self.max_seq_len - len(sample[:-1]))
            )
            item_ids_fut = torch.tensor([sample[-1]])

        else:
            item_ids = self.sequence_data["itemId"][idx]
            item_ids_fut = self.sequence_data["itemId_fut"][idx]

        assert (item_ids >= -1).all(), "Invalid item id found"
        x_brand_id = torch.Tensor(self.item_brand_id[item_ids])
        x_brand_id[item_ids == -1] = -1.0

        x = self.item_data[item_ids, :768]
        x_fut = self.item_data[item_ids_fut]
        x_fut[item_ids_fut == -1] = -1
        x_image = None
        
        # use pre-computed image features
        if self.use_image_features and self.image_features is not None:
            valid_item_mask = item_ids >= 0
            if valid_item_mask.any():
                image_features = self.image_features[item_ids[valid_item_mask]].to(x.device)
                # convert image features to same dtype as x
                image_features = image_features.to(x.dtype)
                
                if self.feature_combination_mode == "sum":
                    x[valid_item_mask] = x[valid_item_mask] + image_features
                    
                elif self.feature_combination_mode == "concat":
                    seq_len, x_dim = x.shape
                    img_dim = image_features.shape[1]
                    # allocate new tensor
                    x_new = torch.zeros(seq_len, x_dim + img_dim, device=x.device, dtype=x.dtype)
                    # copy valid x
                    x_valid = x[valid_item_mask]
                    x_new[valid_item_mask, :x_dim] = x_valid
                    # append image features only to valid rows
                    x_new[valid_item_mask, x_dim:] = image_features
                    x = x_new
                    
                    # ⭐ FIX: Also populate x_image for the tokenizer/model
                    x_image = torch.zeros(self.max_seq_len, img_dim, device=x.device, dtype=x.dtype)
                    valid_idx = valid_item_mask.nonzero(as_tuple=True)[0]
                    if len(valid_idx) > 0:
                        x_image[valid_idx] = image_features
                    
                elif self.feature_combination_mode == "cross-attn":
                    x_image = torch.zeros(self.max_seq_len, image_features.shape[1], device=x.device, dtype=x.dtype)
                    # positions of valid items
                    valid_idx = valid_item_mask.nonzero(as_tuple=True)[0]
                    if len(valid_idx) > 0:
                        # copy image features into the correct positions
                        x_image[valid_idx] = image_features
                else:
                    raise ValueError("Invalid feature combination mode!")
        
        # masking invalid ids    
        x[item_ids == -1] = -1

        return SeqBatch(
            user_ids=user_ids,
            ids=item_ids,
            ids_fut=item_ids_fut,
            x=x,
            x_image=x_image,
            x_brand_id=-1 * torch.ones_like(item_ids.squeeze(0)),
            x_fut=x_fut,
            x_fut_brand_id=-1 * torch.ones_like(item_ids.squeeze(0)),
            seq_mask=(item_ids >= 0),
        )


if __name__ == "__main__":
    dataset = ItemData(
        "dataset/amazon", dataset=RecDataset.AMAZON, split="beauty", force_process=True
    )
    dataset[0]
    train_dataset = SeqData(
        root="dataset/amazon",
        dataset=RecDataset.AMAZON,
        is_train=True,
        subsample=True,
        split="beauty",
    )
    print("train_dataset", train_dataset[0])
    eval_dataset = SeqData(
        root="dataset/amazon",
        dataset=RecDataset.AMAZON,
        is_train=False,
        subsample=False,
        split="beauty",
        get_brand_id=True,
    )
    print("eval_dataset", eval_dataset[0])
    import pdb

    pdb.set_trace()