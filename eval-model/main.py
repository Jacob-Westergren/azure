import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from pose_format.torch.masked.collator import zero_pad_collator
from model import AutoEncoderLightningWrapper, PoseFSQAutoEncoder
from dataset import AzureDataset, PackedDataset
from pytorch_lightning.loggers import MLFlowLogger

from pose_format import PoseHeader
from pose_format.utils.reader import BufferReader
from pathlib import Path
import mlflow

def load_pose_header(data_dir):
    with open(Path(data_dir) / "poseheader", "rb") as f:
        return PoseHeader.read(BufferReader(f.read()))


def evaluate_model(checkpoint_path: str,
                  data_dir: str,
                  batch_size: int = 32,
                  num_workers: int = 1,
                  device: str = "cpu"):
    """
    Evaluate a trained model on test data.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        data_dir: Directory containing the data
        batch_size: Batch size for evaluation
        device: Type of device to use
    """
    # Load the model
    print("Loading model...")
    print(f"Checkpoint file size: {os.path.getsize(checkpoint_path)} bytes")


    # Try to load the checkpoint directly first
    print("Attempting to load checkpoint directly...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print("Successfully loaded checkpoint directly")
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    print(f"State dict keys: {list(checkpoint['state_dict'].keys())}")
    
    # Load pose header
    print("Loading pose header...")
    header = load_pose_header(data_dir)
    
    # Create model with dummy parameters (will be overwritten by state dict)
    print("Creating model...")
    vqvae = PoseFSQAutoEncoder(
        codebook_size=1024,  # These will be overwritten by state dict
        num_codebooks=1,
        num_layers=1
    )
    
    model = AutoEncoderLightningWrapper(
        model=vqvae,
        learning_rate=0.0001,  # This value doesn't matter for evaluation
        loss_weights=None,
        pose_header=header
    )
    
    # Load the state dict directly
    print("Loading state dict...")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Successfully loaded model from state dict")
    # https://lightning.ai/forums/t/best-way-to-use-load-from-checkpoint-when-model-contains-other-models/2094/2

        
    try:
        mlflow_model = mlflow.pytorch.load_model("models:/pose_autoencoder/@latest")    # hardcoded for simplicity, should prob actually be an output from train component
        mlflow_model.eval()
        print("Successfully loaded mlflow model")
    except Exception as e:
        print(f"Error loading mlflow model: {str(e)}")
        print(f"Error type: {type(e)}")
        
    # Load and split data the same way as training
    print("Creating Dataset...")
    dataset = AzureDataset(data_dir_path=data_dir, max_length=512)
    print(f"Successfully created dataset with {len(dataset)} samples!")
    
    # Split dataset with the same seed as training
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = int(0.1 * len(dataset))    # 10% for validation
    test_size = len(dataset) - train_size - val_size  # 10% for testing

    training_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # same seed as training
    )
    
    print(f"Created test dataset with {len(test_dataset)} samples")
    
    # Create test dataloader
    num_workers = os.cpu_count()
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=zero_pad_collator
    )
    
    # Try to read the run_id from the file
    model_dir = os.path.dirname(checkpoint_path)
    run_id_path = os.path.join(model_dir, "run_id.txt")
    run_id = None
    if os.path.exists(run_id_path):
        with open(run_id_path, "r") as f:
            run_id = f.read().strip()
            print(f"Found run_id: {run_id}")
    else:
        print("No run_id file found, will create a new MLflow run")
    
    # Setup MLflow logger to use the same run as training
    mlf_logger = MLFlowLogger(run_id=run_id) if run_id else MLFlowLogger()
    precision = "bf16-mixed" # if args.dtype == "bfloat16" else ("16-mixed" if args.dtype == "float16" else None)
    trainer = pl.Trainer(accelerator=device,
                         profiler="simple",
                         precision=precision,
                         logger=mlf_logger
                         )
    
    # Run evaluation
    print("Running evaluation...")
    results = trainer.test(model=model, dataloaders=test_dataloader)
    
    print(f"Evaluation results: {results}")
    print("Evaluation complete")

    # Reuse the MLflow logger or use a new one
    print(f"Testing Resuming training")
    resume_trainer = pl.Trainer(
        accelerator=device,
        val_check_interval=1,
        profiler="simple",
        precision=precision,
        logger=mlf_logger,
        max_steps=25,   # apparantly the checkpoint stuff remembers the current step, so if i have max_step 5 it wont work, cuz it's not like the current step + 5 but actually tries to set the step max to 5 so no further training is done
        gradient_clip_val=1  
    )

    shuffle = True
    training_iter_dataset = PackedDataset(training_dataset, max_length=512, shuffle=shuffle)
    print("Succesfully created the packed dataset")
    train_dataloader = DataLoader(training_iter_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                collate_fn=zero_pad_collator)   # function to ensure each tensor in batch is same length using padding
    print(f"sucessfulyl created train dataloader")
    val_dataloader = DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    collate_fn=zero_pad_collator)
    print(f"sucessfully created val dataloader")

    # Fit from checkpoint
    print("Resuming training...")
    resume_trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=checkpoint_path)
    print("Finished resumed training.")

if __name__ == "__main__":
    import argparse
    print("Evaluating model...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    print(f"args: {args}")
    
    # Debug: List contents of model directory
    print(f"Contents of model directory {args.model_dir}:")
    for file in os.listdir(args.model_dir):
        print(f"  - {file}")
    
    checkpoint_path = os.path.join(args.model_dir, 'model.ckpt')
    print(f"Looking for checkpoint at: {checkpoint_path}")
    print(f"Checkpoint exists: {os.path.exists(checkpoint_path)}")
    
    evaluate_model(
        checkpoint_path=checkpoint_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=args.device
    ) 