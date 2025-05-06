import math
import logging
from typing import List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

class ConvFeatureExtractionModel(nn.Module):
    def __init__(
            self,
            conv_layers: List[Tuple[int, int, int]],
            dropout: float = 0.0,
            mode: str = "default",
            conv_bias: bool = False,
            conv_type: str = "default",
            in_channels: int = 274
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

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
        in_d = in_channels
        if self.conv_type == "default":
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
        elif self.conv_type == "conv2d":
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                (dim, k, stride) = cl

                self.conv_layers.append(
                    torch.nn.Conv2d(in_d, dim, k, stride)
                )
                self.conv_layers.append(torch.nn.ReLU())
                in_d = dim
        elif self.conv_type == "custom":
            idim = 80
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                (dim, k, stride) = cl # channels, kernel width, stride
                self.conv_layers.append(
                    torch.nn.Conv2d(in_d, dim, k, stride, padding=1)
                )
                self.conv_layers.append(
                    torch.nn.LayerNorm([dim, idim])
                )
                self.conv_layers.append(torch.nn.ReLU())
                in_d = dim
                if (i + 1) % 2 == 0:
                    self.conv_layers.append(
                        torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
                    )
                    idim = int(math.ceil(idim / 2))
        else:
            pass

    def forward(self, x, mask=None):
        # BxT -> BxCxT
        #x = x.unsqueeze(1) # for sign language, we don't have 1 channel (as they have mono audio) but we convert num keypoints * dim = channels
        if self.conv_type == "custom":
            for conv in self.conv_layers:
                if isinstance(conv, nn.LayerNorm):  # layer norm wants input in shape BxTxC
                    x = x.transpose(1, 2)       # BxCxT --> BxTxC
                    x = conv(x).transpose(1, 2) # BxTxC --> BxCxT
                else:                               # No norm or batch/group norm
                    x = conv(x)                 
            x = x.transpose(2, 3).contiguous()
            x = x.view(x.size(0), -1, x.size(-1))
        else:
            for conv in self.conv_layers:
                x = conv(x)
            if self.conv_type == "conv2d":
                b, c, t, f = x.size()
                x = x.transpose(2, 3).contiguous().view(b, c * f, t)
        return x
    
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
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
    """
    
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
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

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask


class WavLMConfig:
    def __init__(self, cfg=None):
        self.extractor_mode: str = "default"     # mode for feature extractor. default has a single group norm with d groups in the first conv block, whereas layer_norm has layer norms in every block (meant to use with normalize=True)
        self.encoder_layers: int = 12     # num encoder layers in the transformer

        self.encoder_embed_dim: int = 768     # encoder embedding dimension
        self.encoder_ffn_embed_dim: int = 3072     # encoder embedding dimension for FFN
        self.encoder_attention_heads: int = 12     # num encoder attention heads
        self.activation_fn: str = "gelu"     # activation function to use

        self.layer_norm_first: bool = False     # apply layernorm first in the transformer
        self.conv_feature_layers: str = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"     # string describing convolutional feature extraction layers in form of a python list that contains [(dim, kernel_size, stride), ...]
        self.conv_bias: bool = False     # include bias in conv encoder
        self.feature_grad_mult: float = 1.0     # multiply feature extractor var grads by this

        self.normalize: bool = False  # normalize input to have 0 mean and unit variance during training

        # dropouts
        self.dropout: float = 0.1     # dropout probability for the transformer
        self.attention_dropout: float = 0.1     # dropout probability for attention weights
        self.activation_dropout: float = 0.0     # dropout probability after activation in FFN
        self.encoder_layerdrop: float = 0.0     # probability of dropping a tarnsformer layer
        self.dropout_input: float = 0.0     # dropout to apply to the input (after feat extr)
        self.dropout_features: float = 0.0     # dropout to apply to the features (after feat extr)

        # masking
        self.mask_length: int = 10     # mask length
        self.mask_prob: float = 0.65     # probability of replacing a token with mask
        self.mask_selection: str = "static"     # how to choose mask length
        self.mask_other: float = 0     # secondary mask argument (used for more complex distributions), see help in compute_mask_indicesh
        self.no_mask_overlap: bool = False     # whether to allow masks to overlap
        self.mask_min_space: int = 1     # min space between spans (if no overlap is enabled)

        # channel masking
        self.mask_channel_length: int = 10     # length of the mask for features (channels)
        self.mask_channel_prob: float = 0.0     # probability of replacing a feature with 0
        self.mask_channel_selection: str = "static"     # how to choose mask length for channel masking
        self.mask_channel_other: float = 0     # secondary mask argument (used for more complex distributions), see help in compute_mask_indices
        self.no_mask_channel_overlap: bool = False     # whether to allow channel masks to overlap
        self.mask_channel_min_space: int = 1     # min space between spans (if no overlap is enabled)

        # positional embeddings
        self.conv_pos: int = 128     # number of filters for convolutional positional embeddings
        self.conv_pos_groups: int = 16     # number of groups for convolutional positional embedding

        # relative position embedding
        self.relative_position_embedding: bool = False     # apply relative position embedding
        self.num_buckets: int = 320     # number of buckets for relative position embedding
        self.max_distance: int = 1280     # maximum distance for relative position embedding
        self.gru_rel_pos: bool = False     # apply gated relative position embedding

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


class MultiGroupFeatureExtractor(nn.Module):
    def __init__(
            self,
            conv_layers: List[Tuple[int, int, int]],
            group_dims: dict,
            dropout: float = 0.0,
            mode: str = "default",
            conv_bias: bool = False,
            conv_type: str = "default",
   ):
        super().__init__()
        
        self.group_dims = group_dims
        
        # Create separate feature extractors for each group
        self.feature_extractors = nn.ModuleDict({
            group: ConvFeatureExtractionModel(
                conv_layers=conv_layers,
                dropout=dropout,
                mode=mode,
                conv_bias=conv_bias,
                conv_type=conv_type,
                in_channels=dim * 2  # Multiply by 2 for x,y coordinates
            )
            for group, dim in self.group_dims.items()
        })
        

        
    
    def forward(self, x, mask=None):
        # x shape: [batch, time, keypoints, 2]
        # Split input into groups
        features = {}
        start_idx = 0
        for group, dim in self.group_dims.items():
            # Extract features for this group
            group_features = self.feature_extractors[group](
                x[:, :, start_idx:start_idx + dim, :].reshape(x.size(0), x.size(1), -1)
            )
            features[group] = group_features
            start_idx += dim
            
        return features
        
   

class WavLM(nn.Module):
    def __init__(
        self,
        cfg: WavLMConfig,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        # original
        # self.feature_extractor = ConvFeatureExtractionModel(
        #     conv_layers=feature_enc_layers,
        #     dropout=0.0,
        #     mode=cfg.extractor_mode,
        #     conv_bias=cfg.conv_bias,
        #     in_channels=274
        # )
        # Replace single feature extractor with multi-group extractor
        self.feature_extractor = MultiGroupFeatureExtractor(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
            group_dims = {
                "body": 24, 
                "hands": 42, 
                "face": 74 
            }
        )
        self.group_dims = self.feature_extractor.group_dims
        # original
        # self.post_extract_proj = (
        #    nn.Linear(self.embed, cfg.encoder_embed_dim)
        #    if self.embed != cfg.encoder_embed_dim
        #    else None
        # )
        # Add projection layer for each group
        self.group_projections = nn.ModuleDict({
            group: 
                nn.Linear(self.embed, cfg.encoder_embed_dim) 
                if self.embed != cfg.encoder_embed_dim else None
                for group in self.group_dims.keys()
        })
        # Compute combined embedding dimension
        self.combined_embed_dim = self.embed * len(self.group_dims)

        # Add final projection to combine features
        self.final_projection = nn.Linear(self.combined_embed_dim, cfg.encoder_embed_dim)
        # Guessing I can either use this or set the encoder_embed_dim to the combined_embed_dim

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

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        dictionaries = [None] * len(self.group_dims)
        # Below is used for pretraining
        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.untie_final_proj = cfg.untie_final_proj
        if self.untie_final_proj:
            self.final_proj = nn.Linear(
                cfg.encoder_embed_dim, final_dim * len(dictionaries)
            )
        else:
            self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)


        # Used for pretraining
        if any([d is None for d in dictionaries]):
            #logger.info("cannot find dictionary. assume will be used for fine-tuning")
            pass
        else:
            self.num_classes = [len(d) for d in dictionaries]
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
            )
            nn.init.uniform_(self.label_embs_concat)

    def compute_loss(self, features, mask_indices, cluster_targets):
        """
        Compute the masked prediction loss and cluster prediction loss.
        
        Args:
            features: Dictionary of feature tensors for each group
            mask_indices: Dictionary of boolean tensors indicating masked positions for each group
            cluster_targets: Dictionary of cluster indices for each group
            
        Returns:
            loss: The combined masked prediction and cluster prediction loss
            sample_size: Number of valid samples for loss computation
        """
        if all(mask is None for mask in mask_indices.values()):
            return 0.0, 0

        total_loss = 0.0
        total_samples = 0

        for group in self.group_dims.keys():
            if mask_indices[group] is not None:
                # Get the masked features
                masked_features = features[group][mask_indices[group]]
                
                # Get the target features (original features at masked positions)
                target_features = self.feature_extractor(self.mask_emb[group])[mask_indices[group]]
                
                # Compute L2 loss between predicted and target features
                feature_loss = F.mse_loss(masked_features, target_features, reduction='none')
                
                # Compute cluster prediction loss
                cluster_loss = F.cross_entropy(
                    self.encoder.cluster_heads[group](masked_features),
                    cluster_targets[group][mask_indices[group]]
                )
                
                # Combine losses
                group_loss = feature_loss.mean() + cluster_loss
                
                total_loss += group_loss
                total_samples += feature_loss.numel()

        # Average the loss
        loss = total_loss / len(self.group_dims) if total_samples > 0 else 0.0
        
        return loss, total_samples

    def forward(self, x, padding_mask=None, mask=True):
        """
        Forward pass of the WavLM model.
        
        Args:
            x: Input tensor of shape [batch_size, channels, sequence_length]
            padding_mask: Optional padding mask
            mask: Whether to apply masking during forward pass
            
        Returns:
            Tuple of (features, padding_mask, loss, sample_size)
        """
        # Extract features
        features, padding_mask = self.extract_features(
            source=x,
            padding_mask=padding_mask,
            mask=mask,
            ret_conv=False,
            output_layer=None,
            ret_layer_results=False
        )
        
        # Compute loss if masking was applied
        loss = 0.0
        sample_size = 0
        if mask and self.mask_prob > 0:
            # Get mask indices from the last extract_features call
            mask_indices = self.last_mask_indices
            loss, sample_size = self.compute_loss(features, mask_indices, padding_mask)
        
        return features, padding_mask, loss, sample_size

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
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        return x, mask_indices

    def forward_padding_mask(
            self, features: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
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
        print(f"Extracting Features...")
        if self.feature_grad_mult > 0:
            group_features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                for group in group_features:
                    group_features[group] = GradMultiply.apply(group_features[group], self.feature_grad_mult)
        else:
            with torch.no_grad():
                group_features = self.feature_extractor(source)

        # Feature penalty punishing large values in the features applied in function get_extra_losses
        # This was not used in the original WavLM paper but was in HuBert paper so unsure if it is needed
        features_pen = 0
        for group in group_features:
            features_pen += group_features[group].float().pow(2).mean()

        X = {}
        mask_indices = {}
        cluster_targets = {}
        for feature in group_features:
            print(f"Feature: {feature.shape}")
            feature = feature.transpose(1, 2)
            feature = self.layer_norm(feature)
            print(f"Layer norm feature: {feature.shape}")

            if padding_mask is not None:
                padding_mask = self.forward_padding_mask(feature, padding_mask)
            
            if self.group_projections[group] is not None:
                feature = self.group_projections[group](feature)
            print(f"Projection feature: {feature.shape}")

            # Compute clusters for this group
            from kmeans_pytorch import kmeans
            num_clusters = 10
            cluster_ids_x, cluster_centers = kmeans(
                X=feature[0], num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
            )
            print(f"Cluster IDs: {cluster_ids_x.shape}")
            print(f"Cluster Centers: {cluster_centers.shape}")
            
            cluster_targets[group] = cluster_ids_x

            if mask:
                x, group_mask = self.apply_mask(feature, padding_mask)
                mask_indices[group] = group_mask
            else:
                x = feature
                mask_indices[group] = None
                
            print(f"{group} x: {x.shape}")
            X[group] = x

        # dont know what shape I should use here, so storing both for now
        # torch.cat([A, B], dim=0) will be of shape (6, 4)
        # torch.stack([A, B], dim=0) will be of shape (2, 3, 4)
        # x = torch.stack([X[group] for group in X], dim=-1)
        # Combine features by concatenating along feature dimension
        # Input shape for each group: [batch_size, sequence_length, embedding_dim]
        # After concatenation: [batch_size, sequence_length, embedding_dim * num_groups]
        x = torch.cat([X[group] for group in X], dim=-1)
        print(f"Combined features shape: {x.shape}")
        


        # Project combined features to transformer dimension
        # x = self.final_projection(x)
        # print(f"Projected features shape: {x.shape}")

        def compute_pred(proj_x, target, label_embs):
            # compute logits for the i-th label set
            y = torch.index_select(label_embs, 0, target.long())
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            # proj_x: (S, D)
            # y: (S, D)
            # negs: (Neg, S, D)
            return self.compute_nce(proj_x, y, negs)

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)

        if not self.skip_masked:
            masked_indices = torch.logical_and(~padding_mask, mask_indices)
            proj_x_m = self.final_proj(x[masked_indices])
            if self.untie_final_proj:
                proj_x_m_list = proj_x_m.chunk(len(target_list), dim=-1)
            else:
                proj_x_m_list = [proj_x_m for _ in range(len(target_list))]
            logit_m_list = [
                compute_pred(proj_x_m, t[masked_indices], label_embs_list[i])
                for i, (proj_x_m, t) in enumerate(zip(proj_x_m_list, target_list))
            ]
        else:
            logit_m_list = [None for _ in target_list]

        if not self.skip_nomask:
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
            proj_x_u = self.final_proj(x[nomask_indices])
            if self.untie_final_proj:
                proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
            else:
                proj_x_u_list = [proj_x_u for _ in range(len(target_list))]

            logit_u_list = [
                compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i])
                for i, (proj_x_u, t) in enumerate(zip(proj_x_u_list, target_list))
            ]
        else:
            logit_u_list = [None for _ in target_list]

        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool

        # Process all features through the transformer
        features, cluster_predictions = self.encoder(
            X,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )
        # Compute loss if masking was applied
        loss = 0.0
        sample_size = 0
        if mask and self.mask_prob > 0:
            loss, sample_size = self.compute_loss(features, mask_indices, cluster_targets)
        return features, padding_mask, loss, sample_size


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

class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.num_clusters = 10  

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        if hasattr(args, "relative_position_embedding"):
            self.relative_position_embedding = args.relative_position_embedding
            self.num_buckets = args.num_buckets
            self.max_distance = args.max_distance
        else:
            self.relative_position_embedding = False
            self.num_buckets = 0
            self.max_distance = 0

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                    has_relative_attention_bias=(self.relative_position_embedding and i == 0),
                    num_buckets=self.num_buckets,
                    max_distance=self.max_distance,
                    gru_rel_pos=args.gru_rel_pos,
                )
                for i in range(args.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        # Cluster prediction heads for each group
        self.cluster_heads = nn.ModuleDict({
            "body": nn.Linear(self.embedding_dim, self.num_clusters),
            "hands": nn.Linear(self.embedding_dim, self.num_clusters),
            "face": nn.Linear(self.embedding_dim, self.num_clusters)
        })

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, streaming_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, streaming_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results
    
    def forward2(self, x_dict, padding_mask=None, streaming_mask=None, layer=None):
        # x_dict is a dictionary of feature vectors for each group
        # Process each group's features through the transformer
        outputs = {}
        for group, x in x_dict.items():
            x, layer_results = self.extract_features(x, padding_mask, streaming_mask, layer)
            if self.layer_norm_first and layer is None:
                x = self.layer_norm(x)
            outputs[group] = x

        # Predict clusters for each group
        cluster_predictions = {}
        for group, x in outputs.items():
            cluster_predictions[group] = self.cluster_heads[group](x)

        return outputs, cluster_predictions

    def extract_features(self, x, padding_mask=None, streaming_mask=None, tgt_layer=None):
        print(f"Transformer: Extracting features...")
        if padding_mask is not None:
            x[padding_mask] = 0
        print(f"Input X: {x.shape}")
        x_conv = self.pos_conv(x.transpose(1, 2))
        print(f"Conv X: {x_conv.shape}")
        x_conv = x_conv.transpose(1, 2)
        print(f"Transposed Conv X: {x_conv.shape}")
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)
            print(f"Layer norm X: {x.shape}")

        x = F.dropout(x, p=self.dropout, training=self.training)
        print(f"Dropout X: {x.shape}")

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        print(f"Transposed X: {x.shape}")

        layer_results = []
        z = None
        if tgt_layer is not None:
            layer_results.append((x, z))
        
        r = None
        pos_bias = None
        print(f"Entering for loop...")
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z, pos_bias = layer(x, self_attn_padding_mask=padding_mask, need_weights=False,
                                       self_attn_mask=streaming_mask, pos_bias=pos_bias)
            if tgt_layer is not None:
                layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break
        print(f"Final X: {x.shape}")
        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        print(f"Transposed back X: {x.shape}")

        return x, layer_results
    
    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
            self,
            embedding_dim: float = 768,
            ffn_embedding_dim: float = 3072,
            num_attention_heads: float = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = "relu",
            layer_norm_first: bool = False,
            has_relative_attention_bias: bool = False,
            num_buckets: int = 0,
            max_distance: int = 0,
            rescale_init: bool = False,
            gru_rel_pos: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            rescale_init=rescale_init,
            gru_rel_pos=gru_rel_pos,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)

        if self.activation_name == "glu":
            self.fc1 = GLU_Linear(self.embedding_dim, ffn_embedding_dim, "swish")
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
            need_weights: bool = False,
            pos_bias=None
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_bias=pos_bias
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
                position_bias=pos_bias
            )
            x = self.dropout1(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn, pos_bias

if __name__ == "__main__":
    batch_size, time_steps, num_keypoints, dim = 1, 512*4, 137, 2
    tensor = torch.rand(batch_size,time_steps,num_keypoints, dim)
    tensor = tensor.view(batch_size, time_steps, num_keypoints*dim)  # [1, 512, 137, 2] --> [1, 512, 274]
    print(tensor.shape)
    tensor = tensor.transpose(1, 2)  # -> [1, 274, 512] for Conv1d as Conv1d expects batch, sequence_length, dim

    wavLMCFG = WavLMConfig(cfg=None)
    wavLM = WavLM(wavLMCFG)
    encoder = wavLM.encoder
    print(wavLM)
    print(encoder)
    print("Input to encoder:", tensor.shape)
    print(f"Output from encoder: {encoder(tensor).shape}")
    wavLM_res = wavLM(tensor)
    print(f"Output from wavLM: {wavLM_res[0].shape}, {wavLM_res[1].shape}")

