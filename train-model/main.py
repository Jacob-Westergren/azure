import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl 
from torch.utils.data import DataLoader, random_split
from pose_format.torch.masked.collator import zero_pad_collator
from dataset import AzureDataset, PackedDataset
import os 
from model import PoseFSQAutoEncoder, AutoEncoderLightningWrapper
from pytorch_lightning.loggers import MLFlowLogger
from pose_format import PoseHeader
from pose_format.utils.reader import BufferReader


def load_pose_header(data_dir):
    with open(Path(data_dir) / "poseheader", "rb") as f:
        return PoseHeader.read(BufferReader(f.read()))

def create_loss_weights(data_dir, hand_weight=1):
    header = load_pose_header(data_dir)

    total_points = header.total_points()
    hand_points = 21
    affected_points = 2 * (hand_points + 1)  # wrist + hand_points
    # We want the loss to be the same scale across different runs, so we change the default weight accordingly
    default_weight = total_points / ((total_points - affected_points) + (affected_points * hand_weight))

    weights = torch.full((total_points, 1), fill_value=default_weight, dtype=torch.float32)
    for hand in ["RIGHT", "LEFT"]:
        # pylint: disable=protected-access
        wrist_index = header._get_point_index(f"{hand}_HAND_LANDMARKS", "WRIST")
        weights[wrist_index: wrist_index + hand_points, :] = hand_weight
        # pylint: disable=protected-access
        body_wrist_index = header._get_point_index("POSE_LANDMARKS", f"{hand}_WRIST")
        weights[body_wrist_index, :] = hand_weight
    return weights


def test_job_2():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--max_pose_length", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--codebook_size", type=int)
    parser.add_argument("--num_codebooks", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--loss_hand_weight", type=float)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--test_data_dir", type=str)  # New argument for test data directory
    args = parser.parse_args()
    print(f"Args:\n{args}")

    torch.set_float32_matmul_precision("medium")

    print("Creating Dataset...")
    dataset = AzureDataset(data_dir_path=args.data_dir, max_length=512)
    print(f"Succesfully created dataset with {len(dataset)} samples!")
    
    # Randomly split dataset into training, validation, and test datasets
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = int(0.1 * len(dataset))    # 10% for validation
    test_size = len(dataset) - train_size - val_size  # 10% for testing

    training_dataset, validation_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )
    
    print(f"Split dataset into {len(training_dataset)} training, {len(validation_dataset)} validation, and {len(test_dataset)} test samples")
    
    shuffle = True
    num_workers = os.cpu_count()

    training_iter_dataset = PackedDataset(training_dataset, max_length=args.max_pose_length, shuffle=shuffle)
    print("Succesfully created the packed dataset")
    train_dataloader = DataLoader(training_iter_dataset,
                                batch_size=args.batch_size,
                                num_workers=num_workers,
                                collate_fn=zero_pad_collator)   # function to ensure each tensor in batch is same length using padding
    print(f"sucessfulyl created train dataloader")
    validation_dataloader = DataLoader(validation_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    collate_fn=zero_pad_collator)
    print(f"sucessfully created val dataloader")

    header = load_pose_header(args.data_dir)
    print(f"header:\n{header}\n")

    # look into save hyperpamaters in the model while ignoring pose_header and loss_weights, and then here i can just do chkpt[hparam][codebook_size]
    auto_encoder = PoseFSQAutoEncoder(codebook_size=args.codebook_size,
                                      num_codebooks=args.num_codebooks,
                                      num_layers=args.num_layers)
    print(f"Successfully created encoder!")
    # loss_weights = create_loss_weights(args.data_dir, hand_weight=args.loss_hand_weight)  fake_pose generates openpose poses, so hand weights algorithm is wrong just ignoring it for now
    print(f"Sucessfully created loss weights!")
    model = AutoEncoderLightningWrapper(auto_encoder, learning_rate=args.lr, loss_weights=None, pose_header=header) #loss_weights)  <-- look above for why
    print(f"Succesfully created model!")


    # Here i decide what precision the tensors should be in, and it needs to match the data in the data asset, or be casted to that during loss eval
    precision = "16-mixed" # aka float16   "bf16-mixed" # if args.dtype == "bfloat16" else ("16-mixed" if args.dtype == "float16" else None)
    # testa precision = torch.float16

    # Setup logger
    mlf_logger = MLFlowLogger() # azureML automatically handles the backend (experiment_name + log dir)

    trainer = pl.Trainer(max_steps=args.steps,
                         val_check_interval=1,  # how many steps to take before running validation, 1 now for debugging
                         accelerator=args.device,
                         profiler="simple",
                         precision=precision,
                         gradient_clip_val=1,  # Taken from the Llamma 2 paper
                         logger=mlf_logger
                         )
    
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)
    print("Finished Training model.")

    # Save the run_id to a file for the evaluation component
    run_id_path = Path(args.model_dir) / "run_id.txt"
    with open(run_id_path, "w") as f:
        f.write(mlf_logger.run_id)
    print(f"Saved run_id {mlf_logger.run_id} to {run_id_path}")

    import mlflow
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name="pose_autoencoder",
    )
    print(f"Successfully registered new model version.")
    
    # Also save the checkpoint for backward compatibility
    checkpoint_path = Path(args.model_dir) / "model.ckpt"
    print(f"Saving checkpoint to {checkpoint_path}")
    trainer.save_checkpoint(checkpoint_path)

    # Verify the checkpoint was saved correctly
    print(f"Checkpoint file size: {os.path.getsize(checkpoint_path)} bytes")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("Successfully loaded checkpoint after saving")
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        print(f"State dict keys: {list(checkpoint['state_dict'].keys())}")
    except Exception as e:
        print(f"Error verifying checkpoint: {str(e)}")
        raise

if __name__ == "__main__":
    test_job_2() 