from data.schemas import SeqBatch
import logging
from rich import print as rprint
from rich.table import Table
from rich.console import Console
from collections import Counter
import torch

# fetch logger
logger = logging.getLogger("recsys_logger")


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


def batch_to(batch, device):
    return SeqBatch(*[v.to(device) if v is not None and hasattr(v, 'to') else v for _, v in batch._asdict().items()])


def next_batch(dataloader, device):
    batch = next(dataloader)
    return batch_to(batch, device)


def describe_dataloader(dataloader, batch_sampler=None, title="DataLoader Summary"):
    """
    Method to print dataset statistics from a PyTorch DataLoader:
        - Total number of samples
        - Total number of batches
        - Class distribution (if available)
        - Sample data shape and dtype
    """
    console = Console()
    dataset = dataloader.dataset
    total_samples = len(dataset)
    total_batches = len(dataloader)

    table = Table(title=title)

    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Total samples", str(total_samples))
    if batch_sampler is None:
        batch_size = dataloader.batch_size
    else:
        batch_size = getattr(batch_sampler, "batch_size", "Unknown")
    table.add_row(f"Total batches (batch_size={batch_size})", str(total_batches))
    table.add_row(f"Num Workers", str(dataloader.num_workers))

    # class info
    class_info_found = False
    if hasattr(dataset, 'classes'):
        table.add_row("Classes", str(dataset.classes))
        class_info_found = True
    if hasattr(dataset, 'class_to_idx'):
        table.add_row("Class to index mapping", str(dataset.class_to_idx))
        class_info_found = True
    if hasattr(dataset, 'data_dict') and 'label' in dataset.data_dict:
        targets = dataset.data_dict['label']
        if isinstance(targets, tuple):
            targets = list(targets)
        if isinstance(targets, list):
            targets = torch.tensor(targets)
        label_counts = Counter(targets.tolist())
        table.add_row("Label counts", str(dict(label_counts)))
        class_info_found = True
    if not class_info_found:
        table.add_row("Class/Label info", "No class/label info found in dataset attributes.")

    # sample data shape and dtype
    try:
        first_batch = next(iter(dataloader))
        if isinstance(first_batch, (list, tuple)):
            # Show shape of first input and sample label summary
            shape_info = str(first_batch[0].shape)
            batch_len = str(len(first_batch))
            table.add_row("Input sample shape", shape_info)
            table.add_row("Batch Len", batch_len)
        elif isinstance(first_batch, dict):
            table.add_row("Sample keys", str(list(first_batch.keys())))
            for key, value in first_batch.items():
                if hasattr(value, 'shape'):
                    shape = tuple(value.shape)
                else:
                    shape = 'N/A'
                dtype = getattr(value, 'dtype', type(value).__name__)
                table.add_row(f"{key.capitalize()} shape & dtype", str(shape) + f", ({str(dtype)})")
        else:
            table.add_row("Sample", str(type(first_batch)))
    except Exception as e:
        table.add_row("Sample inspection error", str(e))

    console.print(table)