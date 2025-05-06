import argparse
from pathlib import Path
import numpy as np
import os
import mlflow
import torch
import pytorch_lightning as pl 
from torch.utils.data import DataLoader
from pose_format.torch.masked.collator import zero_pad_collator
from dataset import AzureDataset, PackedDataset
import os 

def train(data_dir: Path, 
          model_output: str, 
          epochs: int= 100,
          max_pose_length: int= 512,
          batch_size: int = 512):
    print("Starting")
    dataset = AzureDataset(data_dir_path=data_dir, max_length=512)    # check if this is the correct mnt path
    print(f"Succesfully created dataset")
    training_dataset = dataset.slice(5, None)  
    validation_dataset = dataset.slice(0, 5)   # Amit used first 10 poses, should I randomly slice it instead? Could check BatchSampler maybe as well
    print(f"succesfully split the dataset")
    shuffle = True
    num_workers = os.cpu_count()

    training_iter_dataset = PackedDataset(training_dataset, max_length=max_pose_length, shuffle=shuffle)
    print("Succesfully created the packed dataset")
    train_dataset = DataLoader(training_iter_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                collate_fn=zero_pad_collator)   # function to ensure each tensor in batch is same length using padding
    print(f"sucessfulyl created train dataset")
    validation_dataset = DataLoader(validation_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    collate_fn=zero_pad_collator)
    print(f"sucessfulyl created val dataset")

    

    # Output model file
    #model.save(model_output + "/image_classification_model.h5")

