"""
FaultFormer: Transformer Foundation Model for Vibration-Based Fault Diagnosis
=============================================================================
Architecture based on arXiv:2312.02380 (FaultFormer) and arXiv:2409.17604 (RmGPT).

Signal processing pipeline:
  Raw vibration window (2048 points)
    → PatchEmbedding: split into 32 patches of 64 points, project to dim=256
    → Prepend [CLS] token
    → FaultFormer encoder (6 layers, 8 heads, pre-LN)
    → Head: classification / RUL regression / reconstruction

Usage:
  # Pretraining
  model = FaultFormerPretraining()
  loss = model.masked_reconstruction_loss(x)

  # Fine-tuning (classification)
  backbone = FaultFormerPretraining.load_backbone("models/faultformer_pretrained.pt")
  classifier = FaultFormerClassifier(backbone, num_classes=12)
  classifier.freeze_early_layers(num_frozen=4)

  # RUL estimation
  rul_model = FaultFormerRUL(backbone)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGNAL_LENGTH = 2048      # points per vibration window
PATCH_SIZE = 64           # points per patch
NUM_PATCHES = SIGNAL_LENGTH // PATCH_SIZE   # 32
EMBEDDING_DIM = 256       # token embedding dimension
NUM_HEADS = 8             # attention heads
NUM_LAYERS = 6            # transformer encoder layers
FEEDFORWARD_DIM = 1024    # FFN hidden dimension
DROPOUT = 0.1             # dropout rate
NUM_FAULT_CLASSES = 12    # fault classification output classes
MASK_RATIO = 0.75         # fraction of patches masked during pretraining


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Splits a 1D vibration signal into non-overlapping patches and projects
    each patch to the embedding space.

    Args:
        signal_length: total signal length in samples (default 2048)
        patch_size:    samples per patch (default 64 → 32 patches)
        embedding_dim: output embedding dimension (default 256)
        dropout:       dropout on embeddings
    """

    def __init__(
        self,
        signal_length: int = SIGNAL_LENGTH,
        patch_size: int = PATCH_SIZE,
        embedding_dim: int = EMBEDDING_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        assert signal_length % patch_size == 0, (
            f"signal_length {signal_length} must be divisible by patch_size {patch_size}"
        )
        self.patch_size = patch_size
        self.num_patches = signal_length // patch_size
        self.embedding_dim = embedding_dim

        # Linear projection from patch space to embedding space
        self.projection = nn.Linear(patch_size, embedding_dim)

        # Learnable position embeddings (one per patch + one for [CLS])
        self.position_embeddings = nn.Embedding(self.num_patches + 1, embedding_dim)

        # [CLS] token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, signal_length) — raw vibration window

        Returns:
            embeddings: (batch, num_patches + 1, embedding_dim)
                        index 0 is [CLS], indices 1..num_patches are patch tokens
        """
        B = x.shape[0]

        # Split into patches: (batch, num_patches, patch_size)
        patches = x.view(B, self.num_patches, self.patch_size)

        # Linear projection: (batch, num_patches, embedding_dim)
        patch_embeddings = self.projection(patches)

        # Expand [CLS] token across batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embedding_dim)

        # Concatenate CLS + patch tokens
        tokens = torch.cat([cls_tokens, patch_embeddings], dim=1)  # (B, num_patches+1, D)

        # Add positional embeddings
        positions = torch.arange(self.num_patches + 1, device=x.device)
        tokens = tokens + self.position_embeddings(positions)

        return self.dropout(tokens)


# ---------------------------------------------------------------------------
# Transformer Encoder Block (Pre-LN)
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder layer with pre-LN normalization.
    Pre-LN: LayerNorm applied before attention/FFN (as in GPT-2).
    Improves training stability over post-LN for deep transformers.
    """

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        num_heads: int = NUM_HEADS,
        feedforward_dim: int = FEEDFORWARD_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                (batch, seq_len, embedding_dim)
            key_padding_mask: (batch, seq_len) — True for positions to ignore

        Returns:
            (batch, seq_len, embedding_dim)
        """
        # Pre-LN self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out

        # Pre-LN feedforward with residual
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# FaultFormer (Base Encoder)
# ---------------------------------------------------------------------------

class FaultFormer(nn.Module):
    """
    Base transformer encoder backbone for vibration fault diagnosis.

    Architecture:
      PatchEmbedding → 6 x TransformerEncoderBlock → LayerNorm (output)

    The [CLS] token (index 0) provides a global signal summary used by
    classification and regression heads.

    Args:
        signal_length:   vibration window length in samples
        patch_size:      samples per patch
        embedding_dim:   token embedding dimension
        num_heads:       multi-head attention heads
        num_layers:      number of transformer encoder layers
        feedforward_dim: FFN hidden dimension
        dropout:         dropout probability
    """

    def __init__(
        self,
        signal_length: int = SIGNAL_LENGTH,
        patch_size: int = PATCH_SIZE,
        embedding_dim: int = EMBEDDING_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        feedforward_dim: int = FEEDFORWARD_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.patch_embedding = PatchEmbedding(
            signal_length=signal_length,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm on encoder output
        self.norm = nn.LayerNorm(embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following ViT / BERT conventions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                (batch, signal_length)
            key_padding_mask: (batch, num_patches + 1) — True to mask

        Returns:
            encoded: (batch, num_patches + 1, embedding_dim)
                     Index 0 is the [CLS] token embedding.
        """
        tokens = self.patch_embedding(x)   # (B, num_patches+1, D)

        for layer in self.encoder_layers:
            tokens = layer(tokens, key_padding_mask=key_padding_mask)

        return self.norm(tokens)

    def get_cls_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the [CLS] token embedding for a batch of signals.

        Args:
            x: (batch, signal_length)

        Returns:
            cls_emb: (batch, embedding_dim)
        """
        encoded = self.forward(x)
        return encoded[:, 0, :]   # [CLS] is at position 0

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Pretraining: Masked Patch Reconstruction
# ---------------------------------------------------------------------------

class FaultFormerPretraining(FaultFormer):
    """
    FaultFormer with masked patch reconstruction head for self-supervised pretraining.

    Masking strategy:
      - 75% of patch tokens are randomly replaced with a learnable [MASK] token
      - The model reconstructs the original patch values for masked positions
      - Loss is MSE computed only on masked positions (not visible patches)

    This follows the Masked Autoencoder (MAE) paradigm applied to vibration patches.
    The 75% masking ratio is validated in arXiv:2312.02380.
    """

    def __init__(self, mask_ratio: float = MASK_RATIO, **kwargs):
        super().__init__(**kwargs)
        self.mask_ratio = mask_ratio

        # Learnable [MASK] token to replace masked patches
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, self.embedding_dim)
        )
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Reconstruction head: project embedding back to patch space
        patch_size = self.patch_embedding.patch_size
        self.reconstruction_head = nn.Linear(self.embedding_dim, patch_size)

    def _apply_masking(
        self,
        patch_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly mask mask_ratio fraction of patches.

        Args:
            patch_embeddings: (batch, num_patches, embedding_dim)
                              Note: does NOT include [CLS] token here.

        Returns:
            masked_embeddings: (batch, num_patches, embedding_dim)
                               Masked positions replaced with mask_token.
            mask:              (batch, num_patches) — bool tensor, True = masked
        """
        B, N, D = patch_embeddings.shape
        num_masked = int(self.mask_ratio * N)

        # Random shuffle to determine which patches to mask
        noise = torch.rand(B, N, device=patch_embeddings.device)
        shuffle_indices = noise.argsort(dim=1)          # (B, N) ascending
        mask_indices = shuffle_indices[:, :num_masked]   # first num_masked are masked

        # Build boolean mask
        mask = torch.zeros(B, N, dtype=torch.bool, device=patch_embeddings.device)
        mask.scatter_(1, mask_indices, True)

        # Replace masked positions with learnable mask token
        masked_embeddings = patch_embeddings.clone()
        mask_token = self.mask_token.expand(B, N, D)
        masked_embeddings[mask] = mask_token[mask]

        return masked_embeddings, mask

    def forward_pretraining(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for pretraining: apply masking then reconstruct.

        Args:
            x: (batch, signal_length) — raw vibration signals

        Returns:
            reconstructed: (batch, num_patches, patch_size) — reconstructed patches
            original:      (batch, num_patches, patch_size) — original patches
            mask:          (batch, num_patches) — bool, True = masked position
        """
        B = x.shape[0]
        num_patches = self.patch_embedding.num_patches
        patch_size = self.patch_embedding.patch_size

        # Get original patches for target
        original_patches = x.view(B, num_patches, patch_size)

        # Get patch embeddings (before positional encoding for masking)
        # We apply masking to patch embeddings, then add position embeddings
        raw_patches = original_patches  # (B, num_patches, patch_size)
        patch_embs = self.patch_embedding.projection(raw_patches)  # (B, num_patches, D)

        # Apply masking
        masked_patch_embs, mask = self._apply_masking(patch_embs)

        # Add positional embeddings (positions 1..num_patches, skipping [CLS]=0)
        positions = torch.arange(1, num_patches + 1, device=x.device)
        pos_embs = self.patch_embedding.position_embeddings(positions)  # (num_patches, D)
        masked_patch_embs = masked_patch_embs + pos_embs

        # Prepend [CLS] token with its positional embedding
        cls_token = self.patch_embedding.cls_token.expand(B, -1, -1)
        cls_pos = self.patch_embedding.position_embeddings(
            torch.zeros(1, dtype=torch.long, device=x.device)
        )
        cls_token = cls_token + cls_pos

        tokens = torch.cat([cls_token, masked_patch_embs], dim=1)  # (B, num_patches+1, D)
        tokens = self.patch_embedding.dropout(tokens)

        # Run transformer encoder
        for layer in self.encoder_layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)

        # Decode patch tokens (skip [CLS] at index 0)
        patch_tokens = tokens[:, 1:, :]   # (B, num_patches, D)
        reconstructed = self.reconstruction_head(patch_tokens)   # (B, num_patches, patch_size)

        return reconstructed, original_patches, mask

    def masked_reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE reconstruction loss on masked patches only.

        Args:
            x: (batch, signal_length)

        Returns:
            loss: scalar — MSE averaged over masked patch positions
        """
        reconstructed, original, mask = self.forward_pretraining(x)

        # Normalize targets per-patch (as in MAE paper, helps with scale)
        mean = original.mean(dim=-1, keepdim=True)
        std = original.std(dim=-1, keepdim=True) + 1e-6
        target = (original - mean) / std

        # MSE only on masked positions
        loss = F.mse_loss(reconstructed[mask], target[mask])
        return loss

    def save_backbone(self, path: str):
        """Save only the backbone (FaultFormer base) weights, not the pretraining head."""
        backbone_state = {
            k: v for k, v in self.state_dict().items()
            if not k.startswith("reconstruction_head") and not k.startswith("mask_token")
        }
        torch.save({
            "backbone_state_dict": backbone_state,
            "config": {
                "signal_length": self.patch_embedding.num_patches * self.patch_embedding.patch_size,
                "patch_size": self.patch_embedding.patch_size,
                "embedding_dim": self.embedding_dim,
                "num_layers": self.num_layers,
            }
        }, path)
        print(f"Saved pretrained backbone to {path}")

    @classmethod
    def load_backbone(cls, path: str) -> "FaultFormerPretraining":
        """Load pretrained backbone weights."""
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]
        model = cls(
            signal_length=config["signal_length"],
            patch_size=config["patch_size"],
            embedding_dim=config["embedding_dim"],
            num_layers=config["num_layers"],
        )
        missing, unexpected = model.load_state_dict(
            checkpoint["backbone_state_dict"], strict=False
        )
        print(f"Loaded backbone from {path}")
        if missing:
            print(f"  Missing keys (pretraining head, expected): {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        return model


# ---------------------------------------------------------------------------
# Fine-Tuning Head: Fault Classification
# ---------------------------------------------------------------------------

class FaultFormerClassifier(nn.Module):
    """
    FaultFormer with classification head for fault diagnosis.

    Uses the [CLS] token embedding from the pretrained backbone.

    Fine-tuning strategy:
      - Freeze encoder layers 0-3 (early layers with generic representations)
      - Train encoder layers 4-5 + classification head (task-specific adaptation)

    Args:
        backbone:     pretrained FaultFormerPretraining or FaultFormer instance
        num_classes:  number of fault classes (default 12)
    """

    FAULT_CLASSES = [
        "Normal",
        "Outer-race defect",
        "Inner-race defect",
        "Ball defect",
        "Cage defect",
        "Unbalance",
        "Misalignment",
        "Lubrication defect",
        "Looseness",
        "Cavitation",
        "Gearbox fault",
        "Electrical fault",
    ]

    def __init__(
        self,
        backbone: FaultFormer,
        num_classes: int = NUM_FAULT_CLASSES,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        self.classification_head = nn.Sequential(
            nn.LayerNorm(backbone.embedding_dim),
            nn.Linear(backbone.embedding_dim, num_classes),
        )

        self._init_head()

    def _init_head(self):
        for module in self.classification_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def freeze_early_layers(self, num_frozen: int = 4):
        """
        Freeze the first num_frozen encoder layers (0-indexed).
        Layers num_frozen..num_layers-1 remain trainable.

        Typical setting: freeze layers 0-3, train layers 4-5 + head.
        This preserves generic low-level representations while adapting
        high-level features to the target domain.
        """
        # Freeze patch embedding
        for param in self.backbone.patch_embedding.parameters():
            param.requires_grad = False

        # Freeze early encoder layers
        for i, layer in enumerate(self.backbone.encoder_layers):
            if i < num_frozen:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

        # Always train final norm and head
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
        for param in self.classification_head.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Frozen layers 0-{num_frozen-1}. "
              f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, signal_length)

        Returns:
            logits: (batch, num_classes) — unnormalized class scores
        """
        encoded = self.backbone(x)          # (B, num_patches+1, D)
        cls_emb = encoded[:, 0, :]          # (B, D) — [CLS] token
        logits = self.classification_head(cls_emb)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns class probabilities via softmax.

        Returns:
            probs: (batch, num_classes) — probability per class
        """
        with torch.no_grad():
            logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns predicted class index and confidence score.

        Returns:
            class_idx:  (batch,) — argmax class index
            confidence: (batch,) — probability of predicted class
        """
        probs = self.predict_proba(x)
        confidence, class_idx = probs.max(dim=-1)
        return class_idx, confidence

    def get_cls_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the 256-dim [CLS] embedding for similarity search / explainability.

        Args:
            x: (batch, signal_length)

        Returns:
            cls_emb: (batch, embedding_dim) — L2-normalized embedding
        """
        with torch.no_grad():
            encoded = self.backbone(x)
            cls_emb = encoded[:, 0, :]
        return F.normalize(cls_emb, dim=-1)


# ---------------------------------------------------------------------------
# Head: RUL Regression
# ---------------------------------------------------------------------------

class FaultFormerRUL(nn.Module):
    """
    FaultFormer with regression head for Remaining Useful Life (RUL) estimation.

    Output: RUL fraction in [0, 1] where 1.0 = new/healthy, 0.0 = end of life.
    Convert to days by multiplying by the machine's estimated total service life.

    Reuses the same pretrained backbone as FaultFormerClassifier — both tasks
    can share the backbone for multi-task deployment.

    Args:
        backbone:   pretrained FaultFormer instance
        hidden_dim: hidden layer dimension for the regression head
    """

    def __init__(
        self,
        backbone: FaultFormer,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.backbone = backbone

        self.regression_head = nn.Sequential(
            nn.LayerNorm(backbone.embedding_dim),
            nn.Linear(backbone.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),    # Output in [0, 1]
        )

        self._init_head()

    def _init_head(self):
        for module in self.regression_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, signal_length)

        Returns:
            rul_fraction: (batch, 1) — RUL in [0, 1]
        """
        encoded = self.backbone(x)
        cls_emb = encoded[:, 0, :]
        return self.regression_head(cls_emb)

    def predict_rul(
        self,
        x: torch.Tensor,
        total_life_days: float = 365.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict RUL in days.

        Args:
            x:               (batch, signal_length)
            total_life_days: estimated total service life in days

        Returns:
            rul_days:     (batch,) — estimated remaining days
            rul_fraction: (batch,) — fraction in [0, 1]
        """
        with torch.no_grad():
            rul_fraction = self.forward(x).squeeze(-1)
        rul_days = rul_fraction * total_life_days
        return rul_days, rul_fraction


# ---------------------------------------------------------------------------
# Utility: Model Summary
# ---------------------------------------------------------------------------

def model_summary():
    """Print parameter counts for all FaultFormer variants."""
    base = FaultFormer()
    pretrain = FaultFormerPretraining()
    classifier = FaultFormerClassifier(FaultFormer())
    rul = FaultFormerRUL(FaultFormer())

    print("FaultFormer Model Summary")
    print("=" * 50)
    print(f"Base encoder:            {base.parameter_count():>10,} params")
    print(f"Pretraining model:       {sum(p.numel() for p in pretrain.parameters()):>10,} params")
    print(f"Classifier (12 classes): {sum(p.numel() for p in classifier.parameters()):>10,} params")
    print(f"RUL regression:          {sum(p.numel() for p in rul.parameters()):>10,} params")
    print()
    print(f"Patches per window: {NUM_PATCHES} (patch_size={PATCH_SIZE}, signal={SIGNAL_LENGTH})")
    print(f"Sequence length:    {NUM_PATCHES + 1} (+ [CLS])")
    print(f"Embedding dim:      {EMBEDDING_DIM}")
    print(f"Encoder layers:     {NUM_LAYERS} x (8-head attention + FFN-1024)")
    print(f"Masking ratio:      {MASK_RATIO * 100:.0f}% during pretraining")


if __name__ == "__main__":
    model_summary()

    # Quick forward pass test
    print("\nForward pass test:")
    batch_size = 4
    x = torch.randn(batch_size, SIGNAL_LENGTH)

    pretrain_model = FaultFormerPretraining()
    loss = pretrain_model.masked_reconstruction_loss(x)
    print(f"  Pretraining loss: {loss.item():.4f}")

    classifier = FaultFormerClassifier(FaultFormer())
    logits = classifier(x)
    print(f"  Classifier output shape: {logits.shape}")   # (4, 12)

    rul_model = FaultFormerRUL(FaultFormer())
    rul = rul_model(x)
    print(f"  RUL output shape: {rul.shape}")             # (4, 1)
    print(f"  RUL values (fractions): {rul.squeeze().tolist()}")
