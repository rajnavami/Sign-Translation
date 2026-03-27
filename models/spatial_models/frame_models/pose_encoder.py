import torch
import torch.nn as nn


class PoseTemporalEncoder(nn.Module):
    """
    Temporal transformer encoder for pose keypoint sequences.

    Processes the full pose sequence with self-attention so each
    frame's representation contains context from neighboring frames.
    This captures motion patterns that per-frame linear projection cannot.

    Architecture:
        input_dim → Linear → LayerNorm → TransformerEncoder → d_model output

    Input:  sum_T × input_dim  (frames from all batch videos concatenated)
    Output: sum_T × d_model    (same layout, temporally contextualized)

    Args:
        input_dim: dimension of input pose features (default 208 for holistic)
        d_model:   transformer hidden dimension (default 256)
        n_heads:   number of attention heads (default 4)
        n_layers:  number of transformer layers (default 2)
        dropout:   dropout rate (default 0.1)
    """

    def __init__(
        self,
        input_dim: int = 208,
        d_model:   int = 256,
        n_heads:   int = 4,
        n_layers:  int = 2,
        dropout:   float = 0.1,
    ):
        super().__init__()

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.input_dim = input_dim
        self.d_model   = d_model

        # ── Per-frame projection ──────────────────────────────────────────
        # Projects raw pose features to transformer dimension
        # No temporal context yet — just dimension alignment
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # ── Transformer encoder ───────────────────────────────────────────
        # norm_first=True: pre-norm architecture
        # More stable training than post-norm (original BERT style)
        # Each layer: LayerNorm → Attention → Residual
        #             LayerNorm → FFN → Residual
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,   # FFN hidden = 2× d_model = 512
            dropout=dropout,
            activation='gelu',             # GELU better than ReLU for transformers
            batch_first=True,              # B × T × D convention (not T × B × D)
            norm_first=True,               # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,    # avoid nested tensor issues with masking
        )

    def forward(
        self,
        hand_feat: torch.Tensor,
        lengths:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hand_feat: sum_T × input_dim
                       All frames from all videos in the batch concatenated.
                       e.g. if batch has 3 videos with 50, 60, 45 frames:
                       hand_feat has shape (155, 208)

            lengths:   B  (LongTensor)
                       Number of frames per video.
                       e.g. tensor([50, 60, 45])

        Returns:
            sum_T × d_model
            Same layout as input — each frame now has temporal context.
        """
        device = hand_feat.device
        dtype  = hand_feat.dtype
        B      = len(lengths)

        # ── Step 1: Split into per-video sequences ────────────────────────
        sequences = []
        start     = 0
        for l in lengths:
            l = l.item()
            sequences.append(hand_feat[start: start + l])   # l × input_dim
            start += l

        # ── Step 2: Pad to max length ─────────────────────────────────────
        max_len  = max(l.item() for l in lengths)

        # Padded input tensor
        padded   = torch.zeros(
            B, max_len, self.input_dim,
            device=device, dtype=dtype
        )
        # Padding mask:
        #   True  = this position is PADDING → transformer ignores it
        #   False = this position is REAL    → transformer attends to it
        pad_mask = torch.ones(
            B, max_len,
            device=device, dtype=torch.bool
        )

        for i, (seq, l) in enumerate(zip(sequences, lengths)):
            l = l.item()
            padded[i, :l]    = seq
            pad_mask[i, :l]  = False   # real frames → attend

        # ── Step 3: Project to d_model ────────────────────────────────────
        # Shape: B × max_len × d_model
        out = self.input_proj(padded)

        # ── Step 4: Temporal self-attention ───────────────────────────────
        # Each frame attends to all real frames in the same video
        # Padding frames are masked out via src_key_padding_mask
        # Shape: B × max_len × d_model
        out = self.transformer(
            out,
            src_key_padding_mask=pad_mask,
        )

        # ── Step 5: Unpad and concatenate ─────────────────────────────────
        # Return in the same concatenated layout as input
        result = torch.cat([
            out[i, :l.item()]
            for i, l in enumerate(lengths)
        ])   # sum_T × d_model

        return result


class PoseTemporalEncoderWithVelocity(PoseTemporalEncoder):
    """
    Extended version that also computes frame-to-frame velocity
    before the transformer. Doubles the input to the projection layer.

    Velocity = difference between consecutive frames.
    Explicitly captures motion direction and speed.

    input_dim: dimension of raw pose features (208)
    Internally uses input_dim * 2 (208 + 208 velocity = 416)
    """

    def __init__(
        self,
        input_dim: int = 208,
        d_model:   int = 256,
        n_heads:   int = 4,
        n_layers:  int = 2,
        dropout:   float = 0.1,
    ):
        # Initialize parent with doubled input dim
        super().__init__(
            input_dim=input_dim * 2,   # position + velocity
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.raw_input_dim = input_dim

    def _compute_velocity(self, seq):
        """
        seq: T × D
        Returns: T × D  velocity (frame-to-frame differences)
        Forward difference: v[t] = x[t+1] - x[t]
        Last frame: replicate previous velocity
        """
        vel        = torch.zeros_like(seq)
        vel[:-1]   = seq[1:] - seq[:-1]   # v[t] = x[t+1] - x[t]
        vel[-1]    = vel[-2] if len(vel) > 1 else vel[-1]
        return vel

    def forward(
        self,
        hand_feat: torch.Tensor,
        lengths:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Same signature as parent.
        Internally augments each sequence with velocity before passing
        to the transformer.
        """
        # Add velocity per video (not across video boundaries)
        augmented_sequences = []
        start = 0
        for l in lengths:
            l   = l.item()
            seq = hand_feat[start: start + l]       # l × input_dim
            vel = self._compute_velocity(seq)        # l × input_dim
            augmented_sequences.append(
                torch.cat([seq, vel], dim=-1)        # l × (input_dim * 2)
            )
            start += l

        # Reassemble into concatenated format for parent forward
        augmented = torch.cat(augmented_sequences)   # sum_T × (input_dim * 2)

        # Call parent forward with augmented features
        return super().forward(augmented, lengths)