import random
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from pose_format.torch.masked import MaskedTorch
from pose_format.torch.masked.tensor import MaskedTensor
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm


def preprocess_pose(pose, dtype=torch.float16):
    tensor_data = torch.tensor(pose['data'], dtype=dtype)
    tensor_mask = torch.tensor(pose['mask'], dtype=torch.bool)
    tensor_mask = torch.logical_not(tensor_mask)  # numpy and torch have different mask conventions
    tensor = MaskedTensor(tensor=tensor_data, mask=tensor_mask)

    return tensor

def crop_pose(tensor, max_length: int):
    if max_length is not None:
        offset = random.randint(0, len(tensor) - max_length) \
            if len(tensor) > max_length else 0
        return tensor[offset:offset + max_length]
    return tensor

class PackedDataset(IterableDataset):
    def __init__(self, dataset: Dataset, max_length: int, shuffle=True):
        self.dataset = dataset
        self.max_length = max_length
        self.shuffle = shuffle

    def __iter__(self):
        dataset_len = len(self.dataset)
        datum_idx = 0

        datum_shape = self.dataset[0].shape
        padding_shape = tuple([10] + list(datum_shape)[1:])
        padding = MaskedTensor(tensor=torch.zeros(padding_shape), mask=torch.zeros(padding_shape))

        while True:
            poses = []
            total_length = 0
            while total_length < self.max_length:
                if self.shuffle:
                    datum_idx = random.randint(0, dataset_len - 1)
                else:
                    datum_idx = (datum_idx + 1) % dataset_len

                # Append pose
                pose = self.dataset[datum_idx]
                poses.append(pose)
                total_length += len(pose)

                # Append padding
                poses.append(padding)
                total_length += len(padding)

            concatenated_pose = MaskedTorch.cat(poses, dim=0)[:self.max_length]
            yield concatenated_pose


class AzureDataset(Dataset):
    def __init__(self, data_dir_path, max_length=512):
        self.data_dir_path=Path(data_dir_path)
        self.files = list(self.data_dir_path.glob("*.npz"))
        self.max_length = max_length

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.data_dir_path / self.files[idx]
        pose = np.load(file_path)
        return crop_pose(preprocess_pose(pose), self.max_length)


def benchmark_dataloader(dataset, num_workers: int):
    print(f"{num_workers} workers")
    from torch.utils.data import DataLoader
    from pose_format.torch.masked.collator import zero_pad_collator

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True,
                             collate_fn=zero_pad_collator,
                             num_workers=num_workers)
    for _ in tqdm(islice(data_loader, 200)):
        pass


def benchmark():
    # Benchmark
    datasets = [
        AzureDataset(Path('/scratch/amoryo/poses/normalized'), max_length=512),
    ]

    for dataset in datasets:
        print("Benchmarking", dataset.__class__.__name__)

        print("Benchmarking dataset")
        print(next(iter(dataset)).shape)
        for _ in tqdm(islice(iter(dataset), 500)):
            pass

        print("Benchmarking data loader")
        benchmark_dataloader(dataset, 0)
        benchmark_dataloader(dataset, 1)
        benchmark_dataloader(dataset, 4)
        benchmark_dataloader(dataset, 8)


if __name__ == "__main__":
    benchmark()