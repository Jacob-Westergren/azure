from vector_quantize_pytorch import FSQ

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

# check paper's with codebooks of sign language, and try to find good value and convert it to levels approach
# prob 2^8 - 2^10
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


class WavLMConfig:
    def __init__(self, cfg=None):
        self.extractor_mode: str = "default"     # mode for feature extractor. default has a single group norm with d groups in the first conv block, whereas layer_norm has layer norms in every block (meant to use with normalize=True)
        self.encoder_layers: int = 2  # change backt o 12 later     # num encoder layers in the transformer

        self.encoder_embed_dim: int = 768     # encoder embedding dimension
        self.encoder_ffn_embed_dim: int = 768 #3072     # encoder embedding dimension for FFN
        self.encoder_attention_heads: int = 2 #12     # num encoder attention heads
        self.activation_fn: str = "gelu"     # activation function to use

        self.layer_norm_first: bool = False     # apply layernorm first in the transformer
        # # Modified conv layers for sign language (5x downsampling):
        self.conv_feature_layers = "[(512,3,1)] * 2 + [(512,5,5)] + [(512,3,1)] * 2"
        #self.conv_feature_layers: str = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"     # string describing convolutional feature extraction layers in form of a python list that contains [(dim, kernel_size, stride), ...]
        self.conv_bias: bool = False     # include bias in conv encoder
        self.feature_grad_mult: float = 1.0     # multiply feature extractor var grads by this

        self.normalize: bool = False  # normalize input to have 0 mean and unit variance during training

        # Wav2vec: We use dropout 0.1 in the Transformer, at the output of the feature encoder
        # and the input to the quantization module. Layers are dropped at a rate of 0.05 for BASE and 0.2 for
        # LARGE [22, 12]; there is no layer drop for LV-60k
        # dropouts
        self.dropout: float = 0.1     # dropout probability for the transformer
        self.attention_dropout: float = 0.1     # dropout probability for attention weights
        self.activation_dropout: float = 0.0     # dropout probability after activation in FFN
        self.encoder_layerdrop: float = 0.05     # probability of dropping a tarnsformer layer
        self.dropout_input: float = 0.1     # dropout to apply to the input (after feat extr)
        self.dropout_features: float = 0.1     # dropout to apply to the features (after feat extr)

        # masking
        self.mask_length: int = 5     # mask length
        self.mask_prob: float = 0.65     # probability of replacing a token with mask  # wav2vec: For masking, we sample p = 0.065 of all time-steps to be starting indices
        self.mask_selection: str = "static"     # how to choose mask length
        self.mask_other: float = 0     # secondary mask argument (used for more complex distributions), see help in compute_mask_indicesh
        self.no_mask_overlap: bool = False     # whether to allow masks to overlap
        self.mask_min_space: int = 1     # min space between spans (if no overlap is enabled)

        # positional embeddings
        self.conv_pos: int = 128     # number of filters for convolutional positional embeddings
        self.conv_pos_groups: int = 16     # number of groups for convolutional positional embedding

        # relative position embedding
        self.relative_position_embedding: bool = True     # apply relative position embedding
        self.num_buckets: int = 320     # number of buckets for relative position embedding
        self.max_distance: int = 1280     # maximum distance for relative position embedding
        self.gru_rel_pos: bool = True     # apply gated relative position embedding

        # sign language additions
        self.group_dims = {
            "hand": 42*2,
            "face": 72*2,
            "body": 20*2
        }

        # Wav2Vec 2.0 additions
        self.distractors = 100  # Wav2vec: In the contrastive loss we use K = 100 distractors
        # Values inspired by sign-vq:
        self.codebook_size = 2 ** 12  # maybe have different codebook size for the different groups? since like face keypoints >> body keypoints. Have to check other papers
        self.levels = estimate_levels(self.codebook_size)

        # Loss computation
        self.skip_masked = False
        self.skip_nomask = True

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


class WavLM(nn.Module):
    def __init__(
        self,
        cfg: WavLMConfig,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]
        self.num_groups = len(cfg.group_dims.keys())

        ## Feature Extractor
        # Original from HuBERT
          #self.feature_extractor = ConvFeatureExtractionModel(
          #    conv_layers=feature_enc_layers,                       # [(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2
          #    dropout=0.0,
          #    mode=cfg.extractor_mode,                              # default has a single group norm with d groups in the first conv block
          #    conv_bias=cfg.conv_bias,                              # default = false
          #)
        # Modification based on SHuBERT, where I have 1 ConvFeatureExtractionModule per keypoint group
        self.group_feature_extractor = MultiGroupFeatureExtractor(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
            group_dims=cfg.group_dims
        )
        # Difference from shubert:
        #   They had dino as feature extractor, and got 384 dimensional feature vectors for face and hands, and 14 dim for body
        #   I have 1D CNN as feature extractors, with the same dimensional feature vectors of 512 for all groups

        #   They then used a linear layer to project each group to 256 dimensional feature vectors, which were then concatenated
        #   I have a linear layer to project each group to 512 dimensional feature vectors following hubert, which are then concatenated

        #   They mask for a length of 3 to mask 200 ms
        #   I have higher fps so I mask for 5 frames to mask 200 ms

        ## Projection from Feature Extractor to Transformer
        # Original from HuBERT
          #self.post_extract_proj = (
          #    nn.Linear(self.embed, cfg.encoder_embed_dim)
          #    if self.embed != cfg.encoder_embed_dim
          #    else None
          #)
        # Modification based on SHuBERT: Each channel’s normalized features are then projected through a separate linear layer to 256-dimension each.
        self.group_projections = nn.ModuleDict({
            group:
                nn.Linear(self.embed, cfg.encoder_embed_dim)
                if self.embed != cfg.encoder_embed_dim else None
                for group in cfg.group_dims.keys()
        })

        ## Projection from Transformer to Quantized dim * group amount
        # wav2vec 2.0 quantizes the outputs from the featur encoder (CNN)
        # These quantized vectors can then be used in two locations, as inputs to the transformer or as targets for the loss
        # wav2vec 2.0 seems to have tried both but chose to only use the quantize vectors as targets, thus not use quantized vectors
        # in the encoder. This means that I don't need project_inp, as that was used to project the quantized vectors so they would
        # be in the right shape for the transformer as input. Thus I only use project_p from wav2vec 2.0,
        # which projects the quantized targets to their corresponding location in the transformer context space
        # Original from Wav2Vec 2.0
          #self.project_q = nn.Linear(vq_dim, cfg.encoder_embed_dim)
        # Modification based on SHuBERT
        self.vq_dim = self.embed
        self.group_project_q = nn.ModuleDict({
            group:
              nn.Linear(self.vq_dim, cfg.encoder_embed_dim)
              if self.vq_dim != cfg.encoder_embed_dim else None
              for group in cfg.group_dims.keys()
        })

        ## Quantizer
        # I am using FSQ instead of regular codebooks wav2vec 2.0 used, so it's similar in principle but a bit different
        # I am also using 3 quantizers instead of wav2vec 2.0's single quantizer, as I am taking inspiration from SHuBERT with
        # multiple streams and other vq-vae works where they have found benefits using separate codebooks for each group
        self.fsq = {group: FSQ(cfg.levels, dim=self.vq_dim, num_codebooks=1) for group in cfg.group_dims.keys()}

        ## Final Proj
        # we want a linear layer between the context vectors and the predictions, as this is what will make the quantized predictions
        self.final_proj = nn.Linear(cfg.encoder_embed_dim*self.num_groups, cfg.encoder_embed_dim*self.num_groups)
        print(f"\nencoder_embed_dim: {cfg.encoder_embed_dim*self.num_groups}\n")

        ## Masking

        # Mask whole features
        self.mask_prob = cfg.mask_prob                            # default prob = 0.065
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space
        self.mask_emb = nn.Parameter(                             # A masked feature has its content replaced with this
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()   # wav2vec: replace masked vectors with a trained feature vector
        )
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask

        ## Dropout
        self.dropout_input = nn.Dropout(cfg.dropout_input)        # default prob = 0
        self.dropout_features = nn.Dropout(cfg.dropout_features)  # default prob = 0

        self.feature_grad_mult = cfg.feature_grad_mult            # default = 1

        # Transformer, with the same setup as SHuBERT it seems
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        ## Loss

        # Wav2vec 2.0: 100 distractors
        self.num_distractors = cfg.distractors                    # how many distractors to use in the contrastive loss task

        # Wav2vec 2.0: The temperature in the contrastive loss (Equation 3) is set to κ = 0.1
        self.logit_temp = 0.1

    def forward(self, x, padding_mask=None):
        """
        Forward pass of the WavLM model.

        Args:
            x: Input tensor of shape [batch_size, channels, sequence_length]
            padding_mask: Optional padding mask

        Returns:
            Tuple of (features, padding_mask)
        """
        return self.extract_features(
            source=x,
            padding_mask=padding_mask,
            mask=True,
            ret_conv=False,
            output_layer=None,
            ret_layer_results=False
        )

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

    def compute_nce_org(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits

    def compute_nce(self, x, pos, negs):
        """
        x: predicted features [batch_size, dim]
        pos: positive examples [batch_size, dim]
        negs: negative examples [batch_size, num_negs, dim]
        Returns: logits [batch_size, 1 + num_negs]
        """
        print(f"Compute NCE:")
        neg_is_pos = (pos.unsqueeze(1) == negs).all(-1)
        pos = pos.unsqueeze(1)  # [batch_size, 1, dim]
        targets = torch.cat([pos, negs], dim=1)  # [batch_size, 1 + num_negs, dim]
        print(f"targets: {targets.shape}")
        # Compute cosine similarity between predictions and targets
        logits = torch.cosine_similarity(x.unsqueeze(1).float(), targets.float(), dim=-1)  # [batch_size, 1 + num_negs]
        logits = logits / self.logit_temp

        # If a negative is identical to the positive, set its logit to -inf
        if neg_is_pos.any():
            logits[:, 1:][neg_is_pos] = float("-inf")
        print(f"Returning logits: {logits.shape}")

        return logits

    def comp_indices(self, x, padding_mask):
        B, T, C = x.shape
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
        return mask_indices

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
        print(f"source: {source.shape}")
        if self.feature_grad_mult > 0:
            features = self.group_feature_extractor(source)
            if self.feature_grad_mult != 1.0:
              for group in features:
                    features[group] = GradMultiply.apply(features[group], self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.group_feature_extractor(source)

        features_pen = sum(f.float().pow(2).mean() for f in features.values())

        print("Enter group for loop\n")
        unmasked_features = {}
        masked_indices = {}
        x = {}
        for group in features:
            # Normalize
            print(f"Group: {group}, Features shape: {features[group].shape}")
            features[group] = features[group].transpose(1, 2)
            features[group] = self.layer_norm(features[group])
            unmasked_features[group] = features[group].clone()
            print(f"Group: {group}, Layer Norm Features shape: {features[group].shape}")

            # Add mask to padded indices
            if padding_mask is not None:
                padding_mask = self.forward_padding_mask(features[group], padding_mask)

            # Project feature vectors to the context vector's embedding space
            if self.group_projections[group] is not None:
                features[group] = self.group_projections[group](features[group])
                print(f"Projected features: {features[group].shape}")

            # Apply dropout (10%)
            features[group] = self.dropout_input(features[group]) # will be sent through the transformr
            unmasked_features[group] = self.dropout_features(unmasked_features[group])  # will be sent through the quantizer
            print(f"Applied dropout: {features[group].shape}")

            # Apply the masking, as the transformer needs masked input
            if mask:
                x[group], masked_indices[group] = self.apply_mask(
                  features[group],
                  padding_mask
                )
            else:
                x[group] = features[group]
                masked_indices[group] = None

        # Comebine each group to get a unified vector
        X = torch.cat([x[group] for group in x], dim=-1)
        print(f"x[group]: {x['body'].shape}")
        print(f"combined_X: {X.shape}")

        x, layer_results = self.encoder(
            X,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )
        print(f"transformer output X: {x.shape}")

        # Project the context vectors to the quantized vector's space
        x = self.final_proj(x)
        print(f"final_proj X: {x.shape}")

        # Split x into chunks, where each chunk represents one group's quantized vector.
        x_splits = torch.chunk(x, chunks=self.num_groups, dim=-1)
        print(f"X splits:")
        for split in x_splits:
          print(f"split: {split.shape}")
        x = {group: split for group, split in zip(self.cfg.group_dims.keys(), x_splits)}

        # Compute the predictions
        logit_m_list = []
        for group in features:
            if mask and masked_indices[group] is not None:
                # Quantize the features that is masked when going fron CNN to transformer
                print(f"unmasked_features[group]: {unmasked_features[group].shape}")
                print(f"masked_indices[group]: {masked_indices[group].shape}")
                print(f"unmasked_features[group][masked_indices[group]]: {unmasked_features[group][masked_indices[group]].shape}")
                print(f"unmasked_features[group][masked_indices[group]].unsqueeze(0): {unmasked_features[group][masked_indices[group]].unsqueeze(0).shape}")
                quantized_features[group], _ = self.fsq[group](
                    unmasked_features[group][masked_indices[group]].unsqueeze(0)
                )
                target_q = self.group_project_q[group](quantized_features[group])

                # Get masked predictions
                x_masked = x[group][masked_indices[group]]
                print(f"proj_x: {x[group].shape}")
                print(f"target_q: {target_q.shape}")
                print(f"x_masked: {x_masked.shape}")
                print(f"target_masked: {target_q.squeeze(0).shape}")

                # Sample negatives, so that each masked vector will have 100 distractors
                num_masked = masked_indices[group].sum()
                batch_size, seq_len = target_q.shape[:2]
                neg_indices = torch.randint(
                    0, seq_len,
                    (self.num_distractors, num_masked),
                    device=x_masked.device
                )
                batch_indices = torch.zeros(num_masked, dtype=torch.long, device=x_masked.device)
                print(f"batch_size, seq_length = {batch_size}, {seq_len}")
                print(f"neg_indices: {neg_indices.shape}")
                print(f"batch_indices: {batch_indices.shape}")

                # Get negative samples
                negative_samples = target_q[batch_indices, neg_indices].transpose(0, 1)

                # Compute NCE, where we train to predict high logits for positive samples and low logits for negative samples
                print(f"negative_samples: {negative_samples.shape}")
                logits_m = self.compute_nce(x_masked, target_q.squeeze(0), negative_samples)
                logit_m_list.append(logits_m)
        return {
          "logit_m_list": logit_m_list,
          "padding_mask": padding_mask,
          "features_pen": features_pen,
        }

class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        num_groups = len(args.group_dims.keys())
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim*num_groups,
            self.embedding_dim*num_groups,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos*num_groups * self.embedding_dim))
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
                    embedding_dim=self.embedding_dim*num_groups,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim*num_groups,
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
        self.layer_norm = LayerNorm(self.embedding_dim*3)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, streaming_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, streaming_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

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
                res = layer(x, self_attn_padding_mask=padding_mask, need_weights=False,
                            self_attn_mask=streaming_mask, pos_bias=pos_bias)
                #print(f"res: {type(res)}")
                x, z, pos_bias = res
                #print(f"x: {type(x)}")
                #print(f"z: {type(z)}")
                #print(f"pos_bias: {type(pos_bias)}")
            if tgt_layer is not None: # tgt stands for target probably, as in store results up to target layer Nr.3 ex.
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
        # print(layer_results.shape)

        return x, layer_results


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
            gru_rel_pos: bool = False
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
        modules similar to the original Transformer imlementation.
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
        return x, attn, pos_bias