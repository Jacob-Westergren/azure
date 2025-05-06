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

        # sign language additions
        self.group_dims = {
            "hand": 42*2,
            "face": 72*2,
            "body": 20*2
        }

        # Wav2Vec 2.0 additions
        self.distractors = 10
        self.num_codebooks = 4
        self.codebook_size = 2 ** 12  # maybe have different codebook size for the different groups? since like face keypoints >> body keypoints
        self.levels = estimate_levels(self.codebook_size)

        # Loss computation
        skip_masked = False
        skip_nomasd = False

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

        # Original
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,                       # [(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2
            dropout=0.0,
            mode=cfg.extractor_mode,                              # default has a single group norm with d groups in the first conv block
            conv_bias=cfg.conv_bias,                              # default = false
        )
        self.group_feature_extractor = MultiGroupFeatureExtractor(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
            group_dims=cfg.group_dims
        )

        # original
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )
        self.group_projections = nn.ModuleDict({
            group:
                nn.Linear(self.embed, cfg.encoder_embed_dim)
                if self.embed != cfg.encoder_embed_dim else None
                for group in cfg.group_dims.keys()
        })

        # Prediction projections for each group (for contrastive loss)
        self.group_prediction_proj = nn.ModuleDict({
            group: nn.Linear(cfg.encoder_embed_dim, cfg.encoder_embed_dim)
            for group in cfg.group_dims.keys()
        })
        self.logit_temp = nn.Parameter(torch.ones([]) * 0.1)  # temperature parameter for NCE

        # Mask whole features
        self.mask_prob = cfg.mask_prob                            # default prob = 0.65
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        # Mask channels in feature - not used
        self.mask_channel_prob = cfg.mask_channel_prob            # default prob = 0
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)        # default prob = 0
        self.dropout_features = nn.Dropout(cfg.dropout_features)  # default prob = 0

        self.feature_grad_mult = cfg.feature_grad_mult            # default = 1

        self.mask_emb = nn.Parameter(                             # A masked feature has its content replaced with this
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )
        self.fsq = FSQ(cfg.levels, dim=cfg.encoder_embed_dim, num_codebooks=cfg.num_codebooks)
        self.num_distractors = cfg.distractors

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.num_groups = len(cfg.group_dims.keys())

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
            mask=False,
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
            features = self.group_feature_extractor(source)
            if self.feature_grad_mult != 1.0:
              for group in features:
                    features[group] = GradMultiply.apply(features[group], self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.group_feature_extractor(source)

        print("Enter group for loop")
        quantized_features = {}
        group_features = {}
        mask_indices = None
        for group in features.keys():
            print(f"Group: {group}, Features shape: {features[group].shape}")
            features[group] = features[group].transpose(1, 2)
            features[group] = self.layer_norm(features[group])
            print(f"Group: {group}, Layer Norm Features shape: {features[group].shape}")

            if padding_mask is not None:
                padding_mask = self.forward_padding_mask(features[group], padding_mask)

            if self.group_projections[group] is not None:
                features[group] = self.group_projections[group](features[group])
                print(f"Projected features: {features[group].shape}")

            quantized_features[group], indices = self.fsq(features[group])
            print(f"Quantized features: {quantized_features[group].shape}")

            features[group] = self.dropout_input(features[group])

            if mask:
                features[group], mask_indices = self.apply_mask(
                    features[group], padding_mask
                )
            group_features[group] = features[group]
            
        combined_feature_vector = torch.cat([group_features[group] for group in group_features.keys()], dim=-1)
        print("For loop was successful.")
        print(f"combined_X: {combined_feature_vector.shape}")

        x, layer_results = self.encoder(
            combined_feature_vector,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )
        print(f"transformer output x: {x.shape}")

        # Split the encoder output back into group representations
        splits = x.tensor_split(self.num_groups, dim=-1)  # split into groups
        
        if ret_conv or output_layer is not None or ret_layer_results:
            res = {"x": x, "padding_mask": padding_mask, "features": features, "layer_results": layer_results}
            feature = res["features"] if ret_conv else res["x"]
            if ret_layer_results:
                feature = (feature, res["layer_results"])
            return feature, res["padding_mask"]
            
        # Initialize loss related variables
        logit_m_list = []
        logit_u_list = []
        
        # Process each group for contrastive loss
        group_keys = list(self.cfg.group_dims.keys())
        for i, group in enumerate(group_keys):
            # Project the encoder outputs to prediction space
            proj_x = self.group_prediction_proj[group](splits[i])
            
            # Get the ground truth quantized features (targets)
            target_q = quantized_features[group]
            
            if mask and not getattr(self.cfg, "skip_masked", False):
                # Compute loss for masked positions
                masked_indices = torch.logical_and(~padding_mask, mask_indices)
                if masked_indices.any():
                    # For each masked position, use its quantized value as positive
                    # and other quantized values as negatives
                    proj_x_masked = proj_x[masked_indices]
                    target_masked = target_q[masked_indices]
                    
                    # Sample negative examples from other positions
                    batch_size, seq_len = target_q.shape[:2]    # [1, 15, 768] --> 1, 15
                    neg_indices = torch.randint(0, seq_len, (self.num_distractors, masked_indices.sum()), device=proj_x.device) # [10, num_masked_positions]
                    batch_indices = torch.arange(batch_size, device=proj_x.device).repeat_interleave(masked_indices.sum() // batch_size + 1)[:masked_indices.sum()]
                    
                    # Gather negative examples
                    negative_samples = target_q[batch_indices, neg_indices].transpose(0, 1)  # [masked_positions, num_distractors, dim]
                    
                    # Compute NCE loss
                    logits_m = self.compute_nce(proj_x_masked, target_masked, negative_samples)
                    logit_m_list.append(logits_m)
            else:
                logit_m_list.append(None)
                
            if not getattr(self.cfg, "skip_nomask", False):
                # Compute loss for non-masked positions (if required)
                if mask:
                    nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
                else:
                    nomask_indices = ~padding_mask if padding_mask is not None else torch.ones_like(proj_x[:,:,0], dtype=torch.bool)
                
                if nomask_indices.any():
                    proj_x_nomask = proj_x[nomask_indices]
                    target_nomask = target_q[nomask_indices]
                    
                    # Sample negative examples
                    batch_size, seq_len = target_q.shape[:2]
                    neg_indices = torch.randint(0, seq_len, (self.num_distractors, nomask_indices.sum()), device=proj_x.device)
                    batch_indices = torch.arange(batch_size, device=proj_x.device).repeat_interleave(nomask_indices.sum() // batch_size + 1)[:nomask_indices.sum()]
                    
                    # Gather negative examples
                    negative_samples = target_q[batch_indices, neg_indices].transpose(0, 1)
                    
                    # Compute NCE loss
                    logits_u = self.compute_nce(proj_x_nomask, target_nomask, negative_samples)
                    logit_u_list.append(logits_u)
            else:
                logit_u_list.append(None)
        
        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
            "features_pen": 0.0,  # No feature penalty in this implementation
        }
        return result

    def compute_nce(self, x, pos, negs):
        """
        Compute noise contrastive estimation loss
        
        Args:
            x: predicted features [batch_size, dim]
            pos: positive examples [batch_size, dim]
            negs: negative examples [batch_size, num_negs, dim]
        
        Returns:
            logits: classification logits [batch_size, 1 + num_negs]
        """
        neg_is_pos = (pos.unsqueeze(1) == negs).all(-1)
        pos = pos.unsqueeze(1)  # [batch_size, 1, dim]
        targets = torch.cat([pos, negs], dim=1)  # [batch_size, 1 + num_negs, dim]
        
        # Compute cosine similarity between predictions and targets
        logits = torch.cosine_similarity(x.unsqueeze(1).float(), targets.float(), dim=-1)  # [batch_size, 1 + num_negs]
        logits = logits / self.logit_temp
        
        # If a negative is identical to the positive, set its logit to -inf
        if neg_is_pos.any():
            logits[:, 1:][neg_is_pos] = float("-inf")
            
        return logits

    def get_loss(self, result, reduce=True):
        """
        Calculate loss from the logits returned by extract_features
        
        Args:
            result: dictionary containing logit_m_list and logit_u_list
            reduce: whether to reduce loss to scalar
            
        Returns:
            loss: total loss (masked + unmasked)
            masked_loss: loss for masked positions
            unmasked_loss: loss for unmasked positions
        """
        logit_m_list = result["logit_m_list"]
        logit_u_list = result["logit_u_list"]
        
        # Create target labels (first item is positive, rest are negative)
        # so target is always 0 (pointing to the positive sample)
        masked_loss = 0
        unmasked_loss = 0
        
        for i, (logit_m, logit_u) in enumerate(zip(logit_m_list, logit_u_list)):
            if logit_m is not None:
                target = torch.zeros(logit_m.size(0), dtype=torch.long, device=logit_m.device)
                masked_loss += F.cross_entropy(logit_m, target, reduction="sum" if reduce else "none")
                
            if logit_u is not None:
                target = torch.zeros(logit_u.size(0), dtype=torch.long, device=logit_u.device)
                unmasked_loss += F.cross_entropy(logit_u, target, reduction="sum" if reduce else "none")
        
        # Scale losses
        if reduce:
            masked_samples = sum(logit_m.size(0) if logit_m is not None else 0 for logit_m in logit_m_list)
            if masked_samples > 0:
                masked_loss = masked_loss / masked_samples
                
            unmasked_samples = sum(logit_u.size(0) if logit_u is not None else 0 for logit_u in logit_u_list)
            if unmasked_samples > 0:
                unmasked_loss = unmasked_loss / unmasked_samples
        
        loss = masked_loss + unmasked_loss
        
        return loss, masked_loss, unmasked_loss


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