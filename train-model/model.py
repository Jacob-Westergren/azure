"""forked from https://github.com/lucidrains/vector-quantize-pytorch/blob/master/examples/autoencoder_fsq.py"""
import inspect
import math
import sys
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from pose_format.torch.masked import MaskedTensor
from torch import Tensor, nn
from vector_quantize_pytorch import FSQ

user_command = " ".join(sys.argv)
IS_TESTING = "pytest" in user_command or "unittest" in user_command

def estimate_levels(codebook_size: int):
    # Codebook levels based on https://arxiv.org/pdf/2309.15505.pdf Section 4.1
    levels = {
        2 ** 4: [4, 4],  # Not mentioned in the paper, used for tests
        2 ** 8: [8, 6, 5],
        2 ** 9: [4, 5, 5, 5],  # Not mentioned in the paper
        2 ** 10: [8, 5, 5, 5],
        2 ** 11: [4, 4, 5, 5, 5],  # Not mentioned in the paper
        2 ** 12: [7, 5, 5, 5, 5],
        2 ** 14: [8, 8, 8, 6, 5],
        2 ** 16: [8, 8, 8, 5, 5, 5]
    }
    if codebook_size in levels:
        return levels[codebook_size]

    raise ValueError("Codebook size not supported. Supported sizes are 2^4, 2^8, 2^10, 2^11, 2^12, 2^14, 2^16")


class PositionalEncoding(nn.Module):
    # From https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        embedding = torch.zeros(max_len, 1, d_model)
        embedding[:, 0, 0::2] = torch.sin(position * div_term)
        embedding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', embedding)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PoseFSQAutoEncoder(nn.Module):
    # pylint: disable=too-many-arguments
    def __init__(self, codebook_size: int,
                 pose_dims: tuple = (137, 2),
                 num_codebooks: int = 4,
                 hidden_dim=512,
                 nhead=16,
                 dim_feedforward=2048,
                 num_layers=6):
        super().__init__()

        # Store a dictionary of all arguments passed to the constructor
        self.args_dict = locals()
        del self.args_dict['self']
        del self.args_dict['__class__']

        levels = estimate_levels(codebook_size)

        # Calculate the exact number of codes based on the levels
        self.num_codes = math.prod(levels)
        self.num_codebooks = num_codebooks

        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Dropout(0.15),
            nn.Linear(math.prod(pose_dims), hidden_dim, bias=False),
            PositionalEncoding(d_model=hidden_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                           dim_feedforward=dim_feedforward,
                                           batch_first=True),
                num_layers=num_layers
            )
        )

        self.fsq = FSQ(levels, dim=hidden_dim, num_codebooks=num_codebooks)

        self.decoder = nn.Sequential(
            PositionalEncoding(d_model=hidden_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                           dim_feedforward=dim_feedforward,
                                           batch_first=True),
                num_layers=num_layers
            ),
            nn.Linear(hidden_dim, math.prod(pose_dims)),
            nn.Unflatten(dim=2, unflattened_size=pose_dims)
        )

    def __getattr__(self, item):
        if item == "device":
            return next(self.parameters()).device
        return super().__getattr__(item)

    def quantize(self, x: Tensor):
        x = self.encoder(x)
        _, indices = self.fsq(x)
        return indices

    def unquantize(self, indices: Tensor):
        # (batch, codes) or (batch, codes, codebooks)
        indices = indices.view(len(indices), -1, self.num_codebooks)
        x = self.fsq.indices_to_codes(indices)
        x = self.decoder(x)
        return x

    @torch.compile(disable=IS_TESTING or True)
    def forward(self, batch: Union[MaskedTensor, Tensor]):
        tensor = batch.tensor if isinstance(batch, MaskedTensor) else batch
        x = self.encoder(tensor)
        x, indices = self.fsq(x)
        x = self.decoder(x)
        return x, indices


def masked_loss(loss_type: str,
                tensor1: torch.Tensor,
                tensor2: torch.Tensor,
                confidence: torch.Tensor,
                loss_weights: torch.Tensor = None):
    tensor1 = tensor1.to(torch.float32)
    tensor2 = tensor2.to(torch.float32)
    
    assert tensor1.dtype == tensor2.dtype, "Tensors must have the same dtype"
    assert tensor1.dtype == torch.float32, "Tensors must be float32, or casted"
    difference = tensor1 - tensor2

    if loss_type == 'l1':
        error = torch.abs(difference)
    elif loss_type == 'l2':
        error = torch.pow(difference, 2)
    else:
        raise NotImplementedError()

    masked_error = error * confidence  # confidence is 0 for masked values

    if loss_weights is not None:
        masked_error = masked_error * loss_weights

    return masked_error.mean()

"""
from pose_format.pose_visualizer import PoseVisualizer
from pose_format import Pose
import cv2

def pose_from_data(pose_data: Union[MaskedTensor, Tensor], pose_header):
    from pose_format.numpy import NumPyPoseBody

    if isinstance(pose_data, Tensor):
        pose_data = MaskedTensor(pose_data)

    if pose_data.device != torch.device("cpu"):
        pose_data = pose_data.to(torch.device("cpu"))

    # Add person dimension
    pose_data.tensor = pose_data.tensor.unsqueeze(1)
    pose_data.mask = pose_data.mask.unsqueeze(1)

    if pose_data.dtype != torch.float32:
        pose_data.tensor = pose_data.tensor.to(torch.float32)

    np_data = pose_data.tensor.numpy()
    np_confidence = pose_data.mask.numpy().astype(np.float32).max(-1)
    np_body = NumPyPoseBody(fps=25, data=np_data, confidence=np_confidence)

    pose = Pose(header=pose_header, body=np_body)

    # Resize pose
    new_width = 200
    shift = 1.25
    shift_vec = np.full(shape=(pose.body.data.shape[-1]), fill_value=shift, dtype=np.float32)
    pose.body.data = (pose.body.data + shift_vec) * new_width
    pose.header.dimensions.height = pose.header.dimensions.width = int(new_width * shift * 2)

    return pose

def draw_pose(pose_data: MaskedTensor):
    pose = pose_from_data(pose_data)

    # Draw pose
    visualizer = PoseVisualizer(pose)
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in visualizer.draw()]
    return np.stack(frames)

def draw_original_and_predicted_pose(original: MaskedTensor, predicted: Tensor):
    original = MaskedTensor(original.tensor.cpu(), original.mask.cpu())
    predicted = predicted.cpu()

    # to find the pose length, find the last frame where the confidence is not zero
    frame_confidence = original.mask.numpy().max(-1).max(-1)  # (frames)
    pose_length = frame_confidence.nonzero()[0].max() + 1

    original = original[:pose_length]
    predicted = MaskedTensor(predicted[:pose_length])

    original_video = draw_pose(original)
    predicted_video = draw_pose(predicted)
    return np.concatenate([original_video, predicted_video], axis=2)
"""

# pylint: disable=abstract-method,too-many-ancestors,arguments-differ
class AutoEncoderLightningWrapper(pl.LightningModule):
    def __init__(self, model: PoseFSQAutoEncoder,
                 learning_rate: float = 3e-4,
                 warmup_steps: int = 10000,  # For some reason, this is only 400 steps
                 loss_weights: torch.Tensor = None,
                 pose_header = None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_weights = loss_weights
        self.warmup_steps = warmup_steps
        self.header = pose_header
        self.print = True

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        # Optimizer taken from https://arxiv.org/pdf/2307.09288.pdf
        fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters and 'cuda' in str(self.device)
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.learning_rate,
                                      betas=(0.9, 0.95),
                                      eps=1e-5,
                                      weight_decay=0.1,
                                      fused=fused)

        def warm_decay(step):
            if step < self.warmup_steps:
                return min(step / self.warmup_steps, 1)

            # Don't go below a tenth of the learning rate
            return max(0.1, self.warmup_steps ** 0.5 * step ** -0.5)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warm_decay)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # runs per batch rather than per epoch
                "frequency": 1,
                "monitor": "train_loss",
                "name": "learning_rate"  # used in LearningRateMonitor
            }
        }

    def get_codebook_util(self, indices: Tensor):
        if self.model.num_codebooks == 1:
            codebooks = [indices]
        else:
            codebooks = [indices[:, :, i] for i in range(self.model.num_codebooks)]
        uniques = [codebook.unique().numel() for codebook in codebooks]
        mean_unique = torch.tensor(uniques, dtype=torch.float).mean()
        return mean_unique / self.model.num_codes * 100

    def step(self, x: MaskedTensor):
        batch_size = x.shape[0]
 
        x_hat, indices = self(x)

        print(f"[step] Input tensor shape: {x.tensor.shape}, Mask shape: {x.mask.shape}")
        print(f"[step] Output tensor shape: {x_hat.shape}")
        # Input tensor shape: torch.Size([1, 512, 137, 2]), Mask shape: torch.Size([1, 512, 137, 2])
        # Output tensor shape: torch.Size([1, 512, 137, 2])

        if self.loss_weights is not None and self.loss_weights.device != self.device:
            self.loss_weights = self.loss_weights.to(self.device)

        loss = masked_loss('l2', tensor1=x_hat, tensor2=x.tensor,
                           confidence=x.mask, loss_weights=self.loss_weights)

        # both training_step and validation_step calls step, and the bool self.training tells us which process called this instance
        
        phase = "train" if self.trainer.training else (
            "validation" if self.trainer.validating else "test"
        )

        self.log(f"{phase}_code_utilization", self.get_codebook_util(indices), batch_size=1)
        self.log(f"{phase}_loss", loss, batch_size=batch_size)

        return loss, x_hat

    # Pytorch Lightning calls training_step during training, and validation_step during validation, so we simply 
    def training_step(self, batch, *args, **kwargs):
        print("Train step")
        loss, _ = self.step(batch)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        loss, prediction = self.step(batch)
  
        return loss
    
    def test_step(self, batch, batch_idx):
        print("Test Step")
        loss, _ = self.step(batch)
        return loss
