from pose_format.torch.masked import MaskedTensor

import math
import logging
from typing import List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    MultiheadAttention,
    SamePad,
    init_bert_params,
    get_activation_fn,
    TransposeLast,
    GLU_Linear,
)

class ConvFeatureExtractionModel(nn.Module):
    def __init__(
            self,
            conv_layers: List[Tuple[int, int, int]],
            dropout: float = 0.0,
            mode: str = "default",
            conv_bias: bool = False,
            conv_type: str = "default"
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}
        assert conv_type in {"default", "conv2d", "custom", "sign_language"}

        def block(
                n_in,
                n_out,
                k,
                stride,
                is_layer_norm=False,
                is_group_norm=False,
                conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                           is_layer_norm and is_group_norm
                   ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        self.conv_type = conv_type
        if self.conv_type == "default":
            in_d = 1
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3, "invalid conv definition: " + str(cl)
                (dim, k, stride) = cl

                self.conv_layers.append(
                    block(
                        in_d,
                        dim,
                        k,
                        stride,
                        is_layer_norm=mode == "layer_norm",
                        is_group_norm=mode == "default" and i == 0,
                        conv_bias=conv_bias,
                    )
                )
                in_d = dim
        elif self.conv_type == "sign_language":
            # For sign language, we expect input shape: (batch, time, keypoints, coordinates)
            # We'll use a combination of spatial and temporal convolutions
            in_d = 3  # x, y, z coordinates
            self.spatial_layers = nn.ModuleList()
            self.temporal_layers = nn.ModuleList()
            
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                (dim, k, stride) = cl
                
                # Spatial convolution for keypoint relationships
                self.spatial_layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_d, dim, (1, k), stride=(1, stride)),
                        nn.LayerNorm([dim, 1, 1]),
                        nn.GELU()
                    )
                )
                
                # Temporal convolution for motion patterns
                self.temporal_layers.append(
                    nn.Sequential(
                        nn.Conv2d(dim, dim, (k, 1), stride=(stride, 1)),
                        nn.LayerNorm([dim, 1, 1]),
                        nn.GELU()
                    )
                )
                
                in_d = dim
        else:
            # ... existing code for other conv_types ...
            pass

    def forward(self, x, mask=None):
        if self.conv_type == "sign_language":
            # Input shape: (batch, time, keypoints, coordinates)
            # Reshape to: (batch, coordinates, keypoints, time)
            x = x.permute(0, 3, 2, 1)
            
            for spatial_conv, temporal_conv in zip(self.spatial_layers, self.temporal_layers):
                # Apply spatial convolution
                x = spatial_conv(x)
                # Apply temporal convolution
                x = temporal_conv(x)
            
            # Reshape back to: (batch, time, features)
            x = x.permute(0, 3, 1, 2)
            x = x.reshape(x.size(0), x.size(1), -1)
            return x
        else:
            # ... existing forward code for other conv_types ...
            pass 

def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    mask_mode: str = "temporal",  # Added: "temporal", "spatial", or "both"
    spatial_dim: Optional[int] = None,  # Added: number of keypoints
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape
        mask_prob: probability for each token to be chosen as start of the span to be masked
        mask_length: length of the mask span
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        mask_mode: type of masking to apply
            temporal = mask time steps
            spatial = mask keypoints
            both = mask both time steps and keypoints
        spatial_dim: number of keypoints (required for spatial masking)
    """
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    if mask_mode == "spatial" and spatial_dim is None:
        raise ValueError("spatial_dim must be provided for spatial masking")

    if mask_mode == "temporal":
        # Original temporal masking logic
        all_num_mask = int(
            mask_prob * all_sz / float(mask_length)
            + np.random.rand()
        )
        all_num_mask = max(min_masks, all_num_mask)

        mask_idcs = []
        for i in range(bsz):
            if padding_mask is not None:
                sz = all_sz - padding_mask[i].long().sum().item()
                num_mask = int(
                    mask_prob * sz / float(mask_length)
                    + np.random.rand()
                )
                num_mask = max(min_masks, num_mask)
            else:
                sz = all_sz
                num_mask = all_num_mask

            if mask_type == "static":
                lengths = np.full(num_mask, mask_length)
            elif mask_type == "uniform":
                lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
            elif mask_type == "normal":
                lengths = np.random.normal(mask_length, mask_other, size=num_mask)
                lengths = [max(1, int(round(x))) for x in lengths]
            elif mask_type == "poisson":
                lengths = np.random.poisson(mask_length, size=num_mask)
                lengths = [int(round(x)) for x in lengths]
            else:
                raise Exception("unknown mask selection " + mask_type)

            if sum(lengths) == 0:
                lengths[0] = min(mask_length, sz - 1)

            if no_overlap:
                mask_idc = []
                def arrange(s, e, length, keep_length):
                    span_start = np.random.randint(s, e - length)
                    mask_idc.extend(span_start + i for i in range(length))
                    new_parts = []
                    if span_start - s - min_space >= keep_length:
                        new_parts.append((s, span_start - min_space + 1))
                    if e - span_start - keep_length - min_space > keep_length:
                        new_parts.append((span_start + length + min_space, e))
                    return new_parts

                parts = [(0, sz)]
                min_length = min(lengths)
                for length in sorted(lengths, reverse=True):
                    lens = np.fromiter(
                        (e - s if e - s >= length + min_space else 0 for s, e in parts),
                        np.int,
                    )
                    l_sum = np.sum(lens)
                    if l_sum == 0:
                        break
                    probs = lens / np.sum(lens)
                    c = np.random.choice(len(parts), p=probs)
                    s, e = parts.pop(c)
                    parts.extend(arrange(s, e, length, min_length))
                mask_idc = np.asarray(mask_idc)
            else:
                min_len = min(lengths)
                if sz - min_len <= num_mask:
                    min_len = sz - num_mask - 1

                mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)
                mask_idc = np.asarray(
                    [
                        mask_idc[j] + offset
                        for j in range(len(mask_idc))
                        for offset in range(lengths[j])
                    ]
                )

            mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    elif mask_mode == "spatial":
        # Spatial masking: mask keypoints
        num_keypoints = spatial_dim
        all_num_mask = int(mask_prob * num_keypoints + np.random.rand())
        all_num_mask = max(min_masks, all_num_mask)

        mask_idcs = []
        for i in range(bsz):
            keypoint_mask = np.random.choice(num_keypoints, all_num_mask, replace=False)
            # Expand mask to cover all time steps for these keypoints
            mask_idc = np.array([k + t * num_keypoints for k in keypoint_mask for t in range(all_sz // num_keypoints)])
            mask_idcs.append(mask_idc)

    elif mask_mode == "both":
        # Combined temporal and spatial masking
        temporal_mask = compute_mask_indices(
            shape, padding_mask, mask_prob, mask_length, mask_type,
            mask_other, min_masks, no_overlap, min_space, "temporal"
        )
        spatial_mask = compute_mask_indices(
            shape, padding_mask, mask_prob, mask_length, mask_type,
            mask_other, min_masks, no_overlap, min_space, "spatial", spatial_dim
        )
        return np.logical_or(temporal_mask, spatial_mask)

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask 

class PoseEncoder(nn.Module):
    def __init__(self, 
                 pose_dims: tuple = (137, 2),
                 hidden_dim=512,
                 nhead=16,
                 dim_feedforward=2048,
                 num_layers=6):
        super().__init__()
        
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

    def forward(self, x: Union[MaskedTensor, Tensor]):
        tensor = x.tensor if isinstance(x, MaskedTensor) else x
        return self.encoder(tensor)

class WavLM(nn.Module):
    def __init__(
        self,
        cfg: WavLMConfig,
        input_type: str = "audio",  # Added: "audio" or "sign_language"
        num_keypoints: Optional[int] = None,  # Added: number of keypoints for sign language
    ) -> None:
        super().__init__()
        logger.info(f"WavLM Config: {cfg.__dict__}")

        self.cfg = cfg
        self.input_type = input_type
        self.num_keypoints = num_keypoints
        
        if input_type == "sign_language" and num_keypoints is None:
            raise ValueError("num_keypoints must be provided for sign language input")

        if input_type == "sign_language":
            # Use PoseEncoder for sign language input
            self.feature_extractor = PoseEncoder(
                pose_dims=(num_keypoints, 2),  # 2 for x,y coordinates
                hidden_dim=cfg.encoder_embed_dim,
                nhead=cfg.encoder_attention_heads,
                dim_feedforward=cfg.encoder_ffn_embed_dim,
                num_layers=cfg.encoder_layers
            )
        else:
            # Original audio feature extraction
            feature_enc_layers = eval(cfg.conv_feature_layers)
            self.embed = feature_enc_layers[-1][0]
            self.feature_extractor = ConvFeatureExtractionModel(
                conv_layers=feature_enc_layers,
                dropout=0.0,
                mode=cfg.extractor_mode,
                conv_bias=cfg.conv_bias,
                conv_type="default"
            )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim and self.input_type != "sign_language"
            else None
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
                mask_mode="both" if self.input_type == "sign_language" else "temporal",
                spatial_dim=self.num_keypoints if self.input_type == "sign_language" else None,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward_padding_mask(
            self, features: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.input_type == "sign_language":
            # For sign language, we need to handle the spatial dimension
            B, T, K, C = features.shape  # batch, time, keypoints, channels
            extra = padding_mask.size(1) % T
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(
                padding_mask.size(0), T, -1
            )
            padding_mask = padding_mask.all(-1)
            return padding_mask
        else:
            # Original audio handling
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(
                padding_mask.size(0), features.size(1), -1
            )
            padding_mask = padding_mask.all(-1)
            return padding_mask

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
        ret_layer_results: bool = False,
    ):
        if self.input_type == "sign_language":
            # Ensure input is in correct shape: (batch, time, keypoints, coordinates)
            if source.dim() == 3:
                # If input is (batch, time, features), reshape to include keypoints
                B, T, F = source.shape
                K = self.num_keypoints
                C = F // K
                source = source.view(B, T, K, C)

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features, padding_mask
            )
        else:
            x = features

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )

        res = {"x": x, "padding_mask": padding_mask, "features": features, "layer_results": layer_results}

        feature = res["features"] if ret_conv else res["x"]
        if ret_layer_results:
            feature = (feature, res["layer_results"])
        return feature, res["padding_mask"] 