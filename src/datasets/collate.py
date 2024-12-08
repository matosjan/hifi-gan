import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    batch_audio = None
    if 'audio' in dataset_items[0].keys():
        batch_audio = torch.stack([item['audio'] for item in dataset_items])
    batch_spgram = torch.stack([item['melspec'] for item in dataset_items]).squeeze(1)
    batch_path = [entry['path'] for entry in dataset_items]

    result_batch = {
        "audio": batch_audio if batch_audio!=None else None,
        "melspec": batch_spgram,
        "path": batch_path
    }

    return result_batch