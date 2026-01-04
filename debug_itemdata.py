#!/usr/bin/env python3
"""
Debug script to test ItemData and identify dataset issues
"""
import sys
sys.path.insert(0, '/gpfs/home1/scur0555/TIGRESS')

import torch
from data.processed import ItemData, RecDataset
from torch.utils.data import DataLoader

print("=" * 60)
print("Testing ItemData Creation")
print("=" * 60)

# Create dataset
train_dataset = ItemData(
    root="dataset/amazon/2014",
    dataset=RecDataset.AMAZON,
    split="beauty",
    train_test_split="train",
    force_process=False,
    use_patch_embeddings=True,
    patch_model_name="sentence-transformers/sentence-t5-xl",
    patch_max_seq_length=77,
)

print(f"✓ Dataset created: {type(train_dataset)}")
print(f"✓ Dataset length: {len(train_dataset)}")
print(f"✓ Has __getitem__: {hasattr(train_dataset, '__getitem__')}")

# Test single item access
print("\n" + "=" * 60)
print("Testing Single Item Access")
print("=" * 60)
try:
    item = train_dataset[0]
    print(f"✓ train_dataset[0] works!")
    print(f"✓ Item type: {type(item)}")
    print(f"✓ Item fields: {item._fields if hasattr(item, '_fields') else 'N/A'}")
    if hasattr(item, 'text_patches') and item.text_patches is not None:
        print(f"✓ text_patches shape: {item.text_patches.shape}")
    if hasattr(item, 'text_masks') and item.text_masks is not None:
        print(f"✓ text_masks shape: {item.text_masks.shape}")
except Exception as e:
    print(f"✗ Error accessing item: {e}")
    import traceback
    traceback.print_exc()

# Test collate function
print("\n" + "=" * 60)
print("Testing Collate Function")
print("=" * 60)

def item_collate_fn(batch):
    """Custom collate function for ItemData"""
    import torch
    from data.schemas import SeqBatch
    
    field_names = batch[0]._fields
    field_values = {field: [] for field in field_names}
    
    for item in batch:
        for field in field_names:
            value = getattr(item, field)
            if value is not None:
                field_values[field].append(value)
    
    batched_fields = {}
    for field, values in field_values.items():
        if values:
            batched_fields[field] = torch.stack(values)
        else:
            batched_fields[field] = None
    
    return SeqBatch(**batched_fields)

try:
    # Manually create a batch
    batch = [train_dataset[i] for i in range(4)]
    print(f"✓ Created manual batch of 4 items")
    
    collated = item_collate_fn(batch)
    print(f"✓ Collate function works!")
    print(f"✓ Collated type: {type(collated)}")
    if hasattr(collated, 'text_patches') and collated.text_patches is not None:
        print(f"✓ Collated text_patches shape: {collated.text_patches.shape}")
except Exception as e:
    print(f"✗ Error in collate: {e}")
    import traceback
    traceback.print_exc()

# Test DataLoader
print("\n" + "=" * 60)
print("Testing DataLoader")
print("=" * 60)

try:
    dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=item_collate_fn,
    )
    print(f"✓ DataLoader created")
    print(f"✓ DataLoader dataset type: {type(dataloader.dataset)}")
    print(f"✓ DataLoader dataset is train_dataset: {dataloader.dataset is train_dataset}")
    
    # Try to get first batch
    batch = next(iter(dataloader))
    print(f"✓ Got first batch!")
    print(f"✓ Batch type: {type(batch)}")
    if hasattr(batch, 'text_patches') and batch.text_patches is not None:
        print(f"✓ Batch text_patches shape: {batch.text_patches.shape}")
    
    print("\n✅ ALL TESTS PASSED!")
    
except Exception as e:
    print(f"✗ DataLoader error: {e}")
    import traceback
    traceback.print_exc()