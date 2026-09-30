import logging

import torch

from torch import nn
from einops import repeat, pack, unpack
from huggingface_hub import PyTorchModelHubMixin

from .tokgen_utils.convolution import Convolution
from .tokgen_utils.encoders import MultiScaledScalarEncoder, LinearEncoder
from .vit_utils.positional_encoding import PositionalEncoding
from .vit_utils.transformer import Transformer
from ..Flayers.Fredformer_backbone import Fredformer_backbone

logger = logging.getLogger(__name__)

# ==================================
# ====       Organization:      ====
# ==================================
# ==== class TokenGeneratorUnit ====
# ==== class ViTUnit            ====
# ==== class Mantis8M           ====
# ==== class FDDM_Plugin        ====
# ==== class Mantis8MWithFDDM   ====
# ==================================


class FDDM_Plugin(nn.Module):
    """Wraps the Fredformer backbone so it can serve as the plug-in branch."""

    def __init__(
        self,
        seq_len,
        num_channels,
        # block_size=8,
        output_feature_dim=256,
        channel_first=False,
        fredformer_config=None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_channels = num_channels
        self.channel_first = channel_first

        default_cfg = dict(
            ablation=0,
            mlp_drop=0.1,
            use_nys=0,
            output=0,
            cf_dim=512,
            cf_depth=6,
            cf_heads=8,
            cf_mlp=512,   # 512 -> 256
            cf_head_dim=32,
            cf_drop=0.1,
            patch_len=48,
            stride=48,
            d_model=512,
            head_dropout=0.1,
            padding_patch='end',
            individual=False,
            revin=True,
            affine=False,
            subtract_last=False,
            target_window=256,
        )
        if fredformer_config:
            default_cfg.update(fredformer_config)

        target_window = default_cfg.pop("target_window", seq_len)
        self.target_window = target_window
        self.fredformer = Fredformer_backbone(
            c_in=self.num_channels,
            context_window=self.seq_len,
            target_window=target_window,
            **default_cfg,
        )
        flattened_dim = self.num_channels * self.target_window
        self.feature_norm = nn.LayerNorm(flattened_dim)
        # self.final_projection = nn.Linear(flattened_dim, output_feature_dim)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Expected tensor of shape (B, C, L) or (B, L, C) for FDDM_Plugin")

        if self.channel_first:
            if x.shape[1] != self.num_channels or x.shape[2] != self.seq_len:
                raise ValueError("channel_first input must be shaped (B, num_channels, seq_len)")
            plugin_input = x
        else:
            if x.shape[1] != self.seq_len or x.shape[2] != self.num_channels:
                raise ValueError("channel_last input must be shaped (B, seq_len, num_channels)")
            plugin_input = x.transpose(1, 2).contiguous()

        freq_features = self.fredformer(plugin_input.contiguous())
        batch = freq_features.shape[0]
        freq_features = freq_features.reshape(batch, -1)
        freq_features = self.feature_norm(freq_features)
        # return self.final_projection(freq_features)
        return freq_features


class TokenGeneratorUnit(nn.Module):
    def __init__(self, hidden_dim, num_patches, patch_window_size, scalar_scales, hidden_dim_scalar_enc,
                 epsilon_scalar_enc):
        super().__init__()
        self.num_patches = num_patches
        # token generator for time series objects
        num_ts_feats = 2  # original ts + its diff
        kernel_size = patch_window_size + \
            1 if patch_window_size % 2 == 0 else patch_window_size
        self.convs = nn.ModuleList([
            Convolution(kernel_size=kernel_size,
                        out_channels=hidden_dim, dilation=1)
            for i in range(num_ts_feats)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(normalized_shape=hidden_dim, eps=1e-5)
            for i in range(num_ts_feats)
        ])

        # token generator for scalar statistics
        if scalar_scales is None:
            scalar_scales = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
        num_scalar_stats = 2  # mean + std
        self.scalar_encoders = nn.ModuleList([
            MultiScaledScalarEncoder(
                scalar_scales, hidden_dim_scalar_enc, epsilon_scalar_enc)
            for i in range(num_scalar_stats)
        ])

        # final token projector
        self.linear_encoder = LinearEncoder(
            hidden_dim_scalar_enc * num_scalar_stats + hidden_dim * (num_ts_feats), hidden_dim)

        # scales each time-series w.r.t. its mean and std
        self.ts_scaler = lambda x: (
            x - torch.mean(x, axis=2, keepdim=True)) / (torch.std(x, axis=2, keepdim=True) + 1e-5)

    def forward(self, x):
        with torch.no_grad():
            # compute statistics for each patch
            x_patched = x.reshape(x.shape[0], self.num_patches, -1)
            mean_patched = torch.mean(x_patched, axis=-1, keepdim=True)
            std_patched = torch.std(x_patched, axis=-1, keepdim=True)
            statistics = [mean_patched, std_patched]

        # for each encoder output is (batch_size, num_sub_ts, hidden_dim_scalar_enc)
        scalar_embeddings = [self.scalar_encoders[i](
            statistics[i]) for i in range(len(statistics))]

        # apply convolution for original ts and its diff
        ts_var_embeddings = []
        # diff
        with torch.no_grad():
            diff_x = torch.diff(x, n=1, axis=2)
            # pad by zeros to have same dimensionality as x
            diff_x = torch.nn.functional.pad(diff_x, (0, 1))
        # dim(bs, hidden_dim, len_ts-patch_window_size-1)
        embedding = self.convs[0](self.ts_scaler(diff_x))
        ts_var_embeddings.append(embedding)
        
        # original ts
        # dim(bs, hidden_dim, len_ts-patch_window_size-1)
        embedding = self.convs[1](self.ts_scaler(x))
        ts_var_embeddings.append(embedding)

        # split ts_var_embeddings into patches
        patched_ts_var_embeddings = []
        for i, embedding in enumerate(ts_var_embeddings):
            embedding = self.layer_norms[i](embedding)
            embedding = embedding.reshape(
                embedding.shape[0], self.num_patches, -1, embedding.shape[2])
            embedding = torch.mean(embedding, dim=2)
            patched_ts_var_embeddings.append(embedding)

        # concatenate diff_x, x, mu and std embeddinga and send them to the linear projector
        x_embeddings = torch.cat([
            torch.cat(patched_ts_var_embeddings, dim=-1),
            torch.cat(scalar_embeddings, dim=-1)
        ], dim=-1)
        x_embeddings = self.linear_encoder(x_embeddings)

        return x_embeddings


class ViTUnit(nn.Module):
    def __init__(self, hidden_dim, num_patches, depth, heads, mlp_dim, dim_head, dropout, device):
        super().__init__()
        self.pos_encoder = PositionalEncoding(
            d_model=hidden_dim, dropout=dropout, max_len=num_patches+1)
        self.cls_token = nn.Parameter(torch.randn(hidden_dim).to(device))
        self.transformer = Transformer(
            hidden_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)
        x_embeddings, ps = pack([cls_tokens, x], 'b * d')
        x_embeddings = self.pos_encoder(
            x_embeddings.transpose(0, 1)).transpose(0, 1)
        x_embeddings = self.transformer(x_embeddings)
        cls_tokens, _ = unpack(x_embeddings, ps, 'b * d')
        return cls_tokens.reshape(cls_tokens.shape[0], -1)


class Mantis8M(
    nn.Module,
    PyTorchModelHubMixin,
    # optionally, you can add metadata which gets pushed to the model card
    library_name="mantis",
    repo_url="https://huggingface.co/paris-noah/Mantis-8M/tree/main",
    pipeline_tag="time-series-foundation-model",
    license="mit",
    tags=["time-series-foundation-model"],
):
    """
    The architecture of Mantis time series foundation model.

    Parameters
    ----------
    seq_len: int, default 512
        The sequence length, i.e., the length of each time series. This model does not support data with non-fixed
        sequence length, please make all the time series to be of a fixed length by resizing or padding.
    hidden_dim: int, default=256
        Size of a patch (token), i.e., what the hidden dimension each patch is projected to. At the same time,
        ``hidden_dim`` corresponds to the dimension of the embedding space.
    num_patches: int, default=32
        Number of patches (tokens).
    scalar_scales: list, default=None
        List of scales used for MultiScaledScalarEncoder in TokenGeneratorUnit. By default, initialized as [1e-4, 1e-3,
        1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4].
    hidden_dim_scalar_enc: int, default=32
        Hidden dimension of a scalar encoder used for MultiScaledScalarEncoder in TokenGeneratorUnit.
    epsilon_scalar_enc: float, default=1.1
        A constant term used to tolerate the computational error in computation of scale weights for
        MultiScaledScalarEncoder in TokenGeneratorUnit.
    transf_depth: int, default=6
        Number of transformer layers used for Transformer in ViTUnit.
    transf_num_heads: int, default=8
        Number of self-attention heads used for Transformer in ViTUnit.
    transf_mlp_dim: int, default=512
        Hidden dimension of the MLP (feed-forward) transformer's part used for Transformer in ViTUnit.
    transf_dim_head: int, default=128
        Hidden dimension of the keys, queries and values used for Transformer in ViTUnit.
    transf_dropout: froat, default=0.1
        Dropout value used for Transformer in ViTUnit.
    device: {'cpu', 'cuda'}, default='cuda'
        On which device the model is located.
    pre_training: bool, default=False
        If True, applies an MLP projector after the ViTUnit, which originally was used to pre-train the model using
        InfoNCE contrastive loss.
    """

    def __init__(self, seq_len=512, hidden_dim=256, num_patches=32, scalar_scales=None, hidden_dim_scalar_enc=32,
                 epsilon_scalar_enc=1.1, transf_depth=6, transf_num_heads=8, transf_mlp_dim=512, transf_dim_head=128,
                 transf_dropout=0.1, device='cuda', pre_training=False):

        super().__init__()
        assert (seq_len % num_patches) == 0, print(
            'Seq_len must be the multiple of num_patches')
        patch_window_size = int(seq_len / num_patches)

        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.scalar_scales = scalar_scales
        self.hidden_dim_scalar_enc = hidden_dim_scalar_enc
        self.epsilon_scalar_enc = epsilon_scalar_enc
        self.seq_len = seq_len
        self.pre_training = pre_training

        self.tokgen_unit = TokenGeneratorUnit(hidden_dim=hidden_dim,
                                              num_patches=num_patches,
                                              patch_window_size=patch_window_size,
                                              scalar_scales=scalar_scales,
                                              hidden_dim_scalar_enc=hidden_dim_scalar_enc,
                                              epsilon_scalar_enc=epsilon_scalar_enc)
        self.vit_unit = ViTUnit(hidden_dim=hidden_dim, num_patches=num_patches, depth=transf_depth,
                                heads=transf_num_heads, mlp_dim=transf_mlp_dim, dim_head=transf_dim_head,
                                dropout=transf_dropout, device=device)

        self.prj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.to(device)

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, x):
        # 通过tokgen_unit生成输入x的嵌入表示
        x_embeddings = self.tokgen_unit(x)
        # 将嵌入表示送入视觉转换器(vit_unit)进行处理
        vit_out = self.vit_unit(x_embeddings)
        # 根据是否处于预训练阶段来决定输出
        # if self.pre_training:
        #     # 如果是预训练阶段，则通过投影层(prj)处理vit的输出
        #     print("注意：当前Mantis8M模型处于预训练阶段，forward输出经过prj投影层处理")
        #     return self.prj(vit_out)
        # else:
            # 如果不是预训练阶段，则直接返回vit的输出
        #print("注意：当前Mantis8M模型不处于预训练阶段，forward输出为vit_unit的直接输出")
        return vit_out


class TemporalTransformerBlock(nn.Module):
    """
    Dual-axis temporal block operating along the patch axis for each channel independently.

    Input / output shape
    --------------------
    x: (B, C, M, P)
        B: batch size
        C: number of channels
        M: number of patches
        P: hidden dimension
    """

    def __init__(self, hidden_dim, num_heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        b, c, m, p = x.shape
        x_temporal = x.reshape(b * c, m, p)

        x_temporal_norm = self.norm1(x_temporal)
        attn_out, _ = self.attn(
            x_temporal_norm,
            x_temporal_norm,
            x_temporal_norm,
            need_weights=False,
        )
        x_temporal = x_temporal + attn_out
        x_temporal = x_temporal + self.mlp(self.norm2(x_temporal))

        return x_temporal.reshape(b, c, m, p)


class ChannelTransformerBlock(nn.Module):
    """
    Dual-axis channel block operating along the channel axis for each patch independently.

    Input / output shape
    --------------------
    x: (B, C, M, P)
    """

    def __init__(self, hidden_dim, num_heads=4, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, channel_mask=None):
        if x.shape[1] == 1:
            return x

        b, c, m, p = x.shape
        x_channel = x.permute(0, 2, 1, 3).reshape(b * m, c, p)

        if channel_mask is None:
            x_channel_norm = self.norm1(x_channel)
            attn_out, _ = self.attn(
                x_channel_norm,
                x_channel_norm,
                x_channel_norm,
                need_weights=False,
            )
            x_channel = x_channel + attn_out
            x_channel = x_channel + self.mlp(self.norm2(x_channel))
        else:
            if channel_mask.dim() != 2:
                raise ValueError("channel_mask must have shape (B, C)")
            if channel_mask.shape != (b, c):
                raise ValueError(f"channel_mask must have shape ({b}, {c})")

            # channel_mask uses True/1 for valid channels and False/0 for invalid channels.
            channel_mask = channel_mask.to(device=x.device, dtype=torch.bool)
            mask = channel_mask.unsqueeze(1).expand(b, m, c).reshape(b * m, c)
            # key_padding_mask is the inverse: True means the key position is ignored.
            key_padding_mask = ~mask

            # Skip all-invalid samples because an all-True key_padding_mask can produce NaNs.
            all_invalid_per_sample = ~channel_mask.any(dim=1)
            if all_invalid_per_sample.any():
                logger.warning(
                    "ChannelTransformerBlock received samples with all channels masked invalid; "
                    "skipping channel updates for those samples to avoid NaNs."
                )

            valid_rows = ~all_invalid_per_sample.unsqueeze(1).expand(b, m).reshape(b * m)
            if valid_rows.any():
                x_channel_valid = x_channel[valid_rows]
                x_channel_norm = self.norm1(x_channel_valid)
                key_padding_mask_valid = key_padding_mask[valid_rows]

                attn_out, _ = self.attn(
                    x_channel_norm,
                    x_channel_norm,
                    x_channel_norm,
                    key_padding_mask=key_padding_mask_valid,
                    need_weights=False,
                )
                x_channel_valid = x_channel_valid + attn_out
                x_channel_valid = x_channel_valid + self.mlp(self.norm2(x_channel_valid))

                x_channel = x_channel.clone()
                x_channel[valid_rows] = x_channel_valid

        x_channel = x_channel.reshape(b, m, c, p).permute(0, 2, 1, 3).contiguous()
        return x_channel


class DualAxisEncoder(nn.Module):
    """
    Alternating temporal/channel encoder for multi-channel token tensors.

    Input / output shape
    --------------------
    x: (B, C, M, P)
    """

    def __init__(
        self,
        hidden_dim,
        num_dual_axis_layers=3,
        temporal_heads=8,
        channel_heads=4,
        temporal_mlp_dim=512,
        channel_mlp_dim=512,
        dropout=0.1,
        use_channel_axis_attention=True,
    ):
        super().__init__()
        self.use_channel_axis_attention = bool(use_channel_axis_attention)
        self.temporal_blocks = nn.ModuleList([
            TemporalTransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=temporal_heads,
                mlp_dim=temporal_mlp_dim,
                dropout=dropout,
            )
            for _ in range(num_dual_axis_layers)
        ])
        self.channel_blocks = nn.ModuleList([
            ChannelTransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=channel_heads,
                mlp_dim=channel_mlp_dim,
                dropout=dropout,
            )
            for _ in range(num_dual_axis_layers)
        ])
        if not self.use_channel_axis_attention:
            for p in self.channel_blocks.parameters():
                p.requires_grad_(False)

    def forward(self, x, channel_mask=None):
        for temporal_block, channel_block in zip(self.temporal_blocks, self.channel_blocks):
            x = temporal_block(x)
            if self.use_channel_axis_attention:
                x = channel_block(x, channel_mask=channel_mask)
        return x


class GatedMeanPooling(nn.Module):
    """
    Gated mean pooling over the channel axis.

    Input / output shape
    --------------------
    x: (B, C, P)
    channel_mask: optional (B, C), where 1 denotes a valid channel.
    output: (B, P)
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, channel_mask=None):
        gate_scores = self.gate(x)

        if channel_mask is not None:
            if channel_mask.dim() != 2:
                raise ValueError("channel_mask must have shape (B, C)")
            mask = channel_mask.unsqueeze(-1).to(dtype=torch.bool, device=x.device)
            gate_scores = gate_scores.masked_fill(~mask, float("-inf"))

            all_invalid = ~mask.any(dim=1, keepdim=True)
            if all_invalid.any():
                gate_scores = torch.where(all_invalid, torch.zeros_like(gate_scores), gate_scores)

        channel_weights = torch.softmax(gate_scores, dim=1)
        pooled = torch.sum(channel_weights * x, dim=1)
        return pooled


class SingleQueryAttentionPooling(nn.Module):
    """
    Single-query attention pooling over the channel axis.

    Input / output shape
    --------------------
    x: (B, C, P)
    channel_mask: optional (B, C), where 1 denotes a valid channel.
    output: (B, P)
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, channel_mask=None):
        b, c, p = x.shape
        query = self.query.expand(b, -1, -1)

        if channel_mask is None:
            attn_out, _ = self.attn(
                self.query_norm(query),
                self.kv_norm(x),
                self.kv_norm(x),
                need_weights=False,
            )
            pooled = query + attn_out
            pooled = pooled + self.mlp(self.norm2(pooled))
            return pooled.squeeze(1)

        if channel_mask.dim() != 2:
            raise ValueError("channel_mask must have shape (B, C)")
        if channel_mask.shape != (b, c):
            raise ValueError(f"channel_mask must have shape ({b}, {c})")

        # channel_mask uses True/1 for valid channels and False/0 for invalid channels.
        channel_mask = channel_mask.to(device=x.device, dtype=torch.bool)
        all_invalid = ~channel_mask.any(dim=1)

        if all_invalid.any():
            logger.warning(
                "SingleQueryAttentionPooling received samples with all channels masked invalid; "
                "returning zero pooled features for those samples to avoid NaNs."
            )

        pooled = torch.zeros(b, 1, p, device=x.device, dtype=x.dtype)
        valid_rows = ~all_invalid
        if valid_rows.any():
            x_valid = x[valid_rows]
            query_valid = query[valid_rows]
            key_padding_mask_valid = ~channel_mask[valid_rows]

            x_valid_norm = self.kv_norm(x_valid)
            attn_out, _ = self.attn(
                self.query_norm(query_valid),
                x_valid_norm,
                x_valid_norm,
                key_padding_mask=key_padding_mask_valid,
                need_weights=False,
            )
            pooled_valid = query_valid + attn_out
            pooled_valid = pooled_valid + self.mlp(self.norm2(pooled_valid))
            pooled[valid_rows] = pooled_valid

        return pooled.squeeze(1)


class MantisDA(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="mantis",
    repo_url="https://huggingface.co/paris-noah/Mantis-8M/tree/main",
    pipeline_tag="time-series-foundation-model",
    license="mit",
    tags=["time-series-foundation-model", "dual-axis", "multivariate-time-series"],
):
    """
    Dual-axis Mantis backbone for single-channel and multi-channel time series.

    The model reuses the original TokenGeneratorUnit and ViTUnit as shared modules:
    1. Shared per-channel token generation: (B, C, L) -> (B, C, M, P)
    2. Dual-axis encoding over patch and channel axes: (B, C, M, P) -> (B, C, M, P)
    3. Shared per-channel ViT readout: (B, C, M, P) -> (B, C, P)
    4. Channel pooling over channels: (B, C, P) -> (B, P)

    When the input is single-channel, the model falls back to the original Mantis8M path
    to preserve behavior and checkpoint compatibility as much as possible.
    """

    def __init__(
        self,
        seq_len=512,
        hidden_dim=256,
        num_patches=32,
        scalar_scales=None,
        hidden_dim_scalar_enc=32,
        epsilon_scalar_enc=1.1,
        transf_depth=6,
        transf_num_heads=8,
        transf_mlp_dim=512,
        transf_dim_head=128,
        transf_dropout=0.1,
        num_input_channels=1,
        num_dual_axis_layers=3,
        temporal_heads=8,
        channel_heads=4,
        temporal_mlp_dim=512,
        channel_mlp_dim=512,
        dropout=0.1,
        use_channel_mask=False,
        use_channel_axis_attention=True,
        channel_pool_type="attention",
        device='cuda',
        pre_training=False,
    ):
        super().__init__()
        assert (seq_len % num_patches) == 0, print(
            'Seq_len must be the multiple of num_patches')
        patch_window_size = int(seq_len / num_patches)

        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.scalar_scales = scalar_scales
        self.hidden_dim_scalar_enc = hidden_dim_scalar_enc
        self.epsilon_scalar_enc = epsilon_scalar_enc
        self.seq_len = seq_len
        self.pre_training = pre_training
        self.num_input_channels = num_input_channels
        self.num_dual_axis_layers = num_dual_axis_layers
        self.use_channel_mask = use_channel_mask
        self.use_channel_axis_attention = bool(use_channel_axis_attention)
        self.channel_pool_type = str(channel_pool_type)

        self.tokgen_unit = TokenGeneratorUnit(
            hidden_dim=hidden_dim,
            num_patches=num_patches,
            patch_window_size=patch_window_size,
            scalar_scales=scalar_scales,
            hidden_dim_scalar_enc=hidden_dim_scalar_enc,
            epsilon_scalar_enc=epsilon_scalar_enc,
        )
        self.dual_axis_encoder = DualAxisEncoder(
            hidden_dim=hidden_dim,
            num_dual_axis_layers=num_dual_axis_layers,
            temporal_heads=temporal_heads,
            channel_heads=channel_heads,
            temporal_mlp_dim=temporal_mlp_dim,
            channel_mlp_dim=channel_mlp_dim,
            dropout=dropout,
            use_channel_axis_attention=self.use_channel_axis_attention,
        )
        self.vit_unit = ViTUnit(
            hidden_dim=hidden_dim,
            num_patches=num_patches,
            depth=transf_depth,
            heads=transf_num_heads,
            mlp_dim=transf_mlp_dim,
            dim_head=transf_dim_head,
            dropout=transf_dropout,
            device=device,
        )
        if self.channel_pool_type in {"attention", "single_query_attention"}:
            self.channel_pool = SingleQueryAttentionPooling(
                hidden_dim=hidden_dim,
                num_heads=channel_heads,
                dropout=dropout,
            )
        elif self.channel_pool_type in {"gate", "gated", "gated_mean", "gate_pooling"}:
            self.channel_pool = GatedMeanPooling(hidden_dim=hidden_dim)
        else:
            raise ValueError(
                f"Unsupported channel_pool_type='{self.channel_pool_type}'. "
                "Expected one of: attention, single_query_attention, gate, gated_mean."
            )

        # Kept for checkpoint compatibility with Mantis8M. It is not used in forward.
        self.prj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.to(device)

    def to(self, device):
        self.device = device
        return super().to(device)

    def get_channel_features(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract per-channel ViT features WITHOUT pooling over the channel axis.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, C, L). Multi-channel time-series input.
        channel_mask : torch.Tensor or None
            Shape (B, C) bool, True = valid channel.

        Returns
        -------
        torch.Tensor
            Shape (B, C, P). Per-channel features, where P = self.hidden_dim.
        """
        if x.ndim != 3:
            raise ValueError("Expected input with shape (B, C, L)")
        if x.shape[-1] != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, but got {x.shape[-1]}")

        b, c, _ = x.shape

        # Fast path for single-channel inputs.
        if c == 1:
            return self.vit_unit(self.tokgen_unit(x)).unsqueeze(1)  # (B, 1, P)

        if channel_mask is not None and not self.use_channel_mask:
            raise ValueError("channel_mask was provided but use_channel_mask is False")

        if channel_mask is not None and self.use_channel_mask:
            if channel_mask.dim() != 2 or channel_mask.shape != (b, c):
                raise ValueError(f"channel_mask must have shape ({b}, {c})")
            channel_mask = channel_mask.to(device=x.device, dtype=torch.bool)

        # Shared per-channel token generation.
        channel_tokens = []
        for channel_idx in range(c):
            x_channel = x[:, channel_idx:channel_idx + 1, :]  # (B, 1, L)
            h_channel = self.tokgen_unit(x_channel)            # (B, M, P)
            channel_tokens.append(h_channel)
        hidden = torch.stack(channel_tokens, dim=1)            # (B, C, M, P)

        # Dual-axis token interaction.
        hidden = self.dual_axis_encoder(
            hidden,
            channel_mask=channel_mask if self.use_channel_mask else None,
        )                                                       # (B, C, M, P)

        # Shared per-channel ViT readout.
        channel_features = []
        for channel_idx in range(c):
            h_channel = hidden[:, channel_idx, :, :]           # (B, M, P)
            z_channel = self.vit_unit(h_channel)               # (B, P)
            channel_features.append(z_channel)
        channel_features = torch.stack(channel_features, dim=1)  # (B, C, P)

        return channel_features

    def load_from_mantis(self, mantis_model):
        """
        Load reusable weights from an existing Mantis8M checkpoint.

        Reused parameters
        -----------------
        - tokgen_unit
        - vit_unit
        - prj (optional, if present on both sides)

        Newly initialized parameters
        ----------------------------
        - dual_axis_encoder
        - channel_pool
        """
        self.tokgen_unit.load_state_dict(mantis_model.tokgen_unit.state_dict())
        self.vit_unit.load_state_dict(mantis_model.vit_unit.state_dict())
        if hasattr(mantis_model, "prj") and hasattr(self, "prj"):
            try:
                self.prj.load_state_dict(mantis_model.prj.state_dict())
            except Exception:
                pass

    def forward(self, x, channel_mask=None):
        if x.ndim != 3:
            raise ValueError("Expected input with shape (B, C, L)")
        if x.shape[-1] != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, but got {x.shape[-1]}")

        b, c, _ = x.shape

        # Fast path for single-channel inputs to preserve original Mantis behavior.
        if c == 1:
            return self.vit_unit(self.tokgen_unit(x))

        # if self.num_input_channels is not None and c > self.num_input_channels:
        #     raise ValueError(
        #         f"Input has {c} channels, which exceeds configured num_input_channels={self.num_input_channels}"
        #     )

        if channel_mask is not None and not self.use_channel_mask:
            raise ValueError("channel_mask was provided but use_channel_mask is False")

        if channel_mask is not None and self.use_channel_mask:
            if channel_mask.dim() != 2 or channel_mask.shape != (b, c):
                raise ValueError(f"channel_mask must have shape ({b}, {c})")
            # channel_mask uses True/1 for valid channels and False/0 for invalid channels.
            channel_mask = channel_mask.to(device=x.device, dtype=torch.bool)

        # Shared per-channel token generation.
        channel_tokens = []
        for channel_idx in range(c):
            x_channel = x[:, channel_idx:channel_idx + 1, :]   # (B, 1, L)
            h_channel = self.tokgen_unit(x_channel)            # (B, M, P)
            channel_tokens.append(h_channel)
        hidden = torch.stack(channel_tokens, dim=1)            # (B, C, M, P)

        # Dual-axis token interaction.
        hidden = self.dual_axis_encoder(
            hidden,
            channel_mask=channel_mask if self.use_channel_mask else None,
        )                                                     # (B, C, M, P)

        # Shared per-channel ViT readout.
        channel_features = []
        for channel_idx in range(c):
            h_channel = hidden[:, channel_idx, :, :]           # (B, M, P)
            z_channel = self.vit_unit(h_channel)               # (B, P)
            channel_features.append(z_channel)
        channel_features = torch.stack(channel_features, dim=1)  # (B, C, P)

        pooled = self.channel_pool(
            channel_features,
            channel_mask=channel_mask if self.use_channel_mask else None,
        )
        return pooled


class Mantis8MWithFDDM(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="mantis",
    repo_url="https://huggingface.co/paris-noah/Mantis-8M/tree/main",
    pipeline_tag="time-series-foundation-model",
    license="mit",
    tags=["time-series-foundation-model"],
):
    """Composite model that fuses Mantis8M embeddings with FDDM frequency cues."""

    def __init__(
        self,
        seq_len=512,
        hidden_dim=256,
        num_patches=32,
        num_channels=1,
        scalar_scales=None,
        hidden_dim_scalar_enc=32,
        epsilon_scalar_enc=1.1,
        transf_depth=6,
        transf_num_heads=8,
        transf_mlp_dim=512,
        transf_dim_head=128,
        transf_dropout=0.1,
        fddm_output_dim=256,
        device='cuda',
        pre_training=False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.pre_training = pre_training

        self.mantis = Mantis8M(
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_patches=num_patches,
            scalar_scales=scalar_scales,
            hidden_dim_scalar_enc=hidden_dim_scalar_enc,
            epsilon_scalar_enc=epsilon_scalar_enc,
            transf_depth=transf_depth,
            transf_num_heads=transf_num_heads,
            transf_mlp_dim=transf_mlp_dim,
            transf_dim_head=transf_dim_head,
            transf_dropout=transf_dropout,
            device=device,
            pre_training=pre_training,
        )

        self.fddm_plugin = FDDM_Plugin(
            seq_len=seq_len,
            num_channels=num_channels,
            output_feature_dim=fddm_output_dim,
            channel_first=True,
        )

        #self.fusion_input_dim = self.mantis.hidden_dim + fddm_output_dim
        #self.output_dim = fusion_dim if fusion_dim is not None else self.fusion_input_dim
        # self.fusion_head = nn.Sequential(
        #     nn.LayerNorm(self.fusion_input_dim),
        #     nn.Dropout(fusion_dropout),
        #     nn.Linear(self.fusion_input_dim, self.output_dim),
        # )
        print("注意本次模型forward 只有mantis_features,没有contact-----------------------------------")
        self.to(device)

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, x):
        mantis_features = self.mantis(x)
        freq_features = self.fddm_plugin(x)
        #fused = torch.cat([mantis_features, freq_features], dim=-1)
        # return self.fusion_head(fused)
        return freq_features


class _LatentSelfAttentionBlock(nn.Module):
    """Pre-Norm self-attention + FFN block operating on latent tokens.

    Input / output shape
    --------------------
    x: (N, L, D)
        N: batch size
        L: number of latent tokens
        D: latent dimension
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm residual
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + attn_out
        # FFN with pre-norm residual
        x = x + self.mlp(self.norm2(x))
        return x


class ChannelSetAggregator(nn.Module):
    """Perceiver-style set aggregator for variable-length channel features.

    Replaces the single-vector channel_pool (e.g. SingleQueryAttentionPooling)
    with a learned latent set that cross-attends to per-channel features,
    followed by latent self-attention and a final projection.

    This reduces the information bottleneck when collapsing (N, C, P) into (N, D)
    by maintaining L learnable latent tokens per sample instead of a single vector.

    Input / output shape
    --------------------
    channel_features: (N, C, P)   — per-channel features from MantisDA
    channel_mask:     (N, C) bool or None   — True = valid channel
    output:           (N, D)

    Parameters
    ----------
    input_dim : int
        Per-channel feature dimension P (MantisDA hidden_dim).
    output_dim : int
        Output instance embedding dimension D.
    num_latents : int
        Number of learnable latent queries L (fixed regardless of C).
    num_heads : int
        Attention heads for cross-attention and latent self-attention.
    num_latent_layers : int
        Number of self-attention layers applied on latent tokens.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 512,
        num_latents: int = 16,
        num_heads: int = 4,
        num_latent_layers: int = 2,
        dropout: float = 0.1,
        slot_gate_init: float | None = None,
        **_unused_kwargs,
    ) -> None:
        super().__init__()
        # Accepted for compatibility only; this variant has no slot gate.
        del slot_gate_init, _unused_kwargs
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_latents = int(num_latents)
        D = self.output_dim

        # Per-channel linear projection: P → D
        self.channel_proj = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, D),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learnable latent queries: (1, L, D)
        self.latents = nn.Parameter(torch.randn(1, num_latents, D) * 0.02)

        # Cross-attention: latents (query) attend to channel tokens (key/value)
        self.cross_attn_norm = nn.LayerNorm(D)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=D,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Latent self-attention blocks
        self.latent_layers = nn.ModuleList([
            _LatentSelfAttentionBlock(dim=D, num_heads=num_heads, dropout=dropout)
            for _ in range(num_latent_layers)
        ])

        # Output projection: flatten latents (N, L*D) → (N, D)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(num_latents * D),
            nn.Linear(num_latents * D, D),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        channel_features: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aggregate per-channel features into a fixed-dimension instance embedding.

        Parameters
        ----------
        channel_features : torch.Tensor
            Shape (N, C, P). Per-channel features from MantisDA encoder.
        channel_mask : torch.Tensor or None
            Shape (N, C) bool, True = valid channel, False = padded/missing.
            When None, all channels are treated as valid.

        Returns
        -------
        torch.Tensor
            Shape (N, D). Instance-level embedding.
        """
        N, C, _P = channel_features.shape

        # --- Step 1: Per-channel projection ---
        x = self.channel_proj(channel_features)  # (N, C, D)

        # --- Step 2: Cross-attention: L latents attend to C channels ---
        latents = self.latents.expand(N, -1, -1)  # (N, L, D)

        all_invalid: torch.Tensor | None = None  # (N,) bool — samples to zero out

        if channel_mask is not None:
            if channel_mask.dim() != 2 or channel_mask.shape != (N, C):
                raise ValueError(f"channel_mask must have shape ({N}, {C}), got {tuple(channel_mask.shape)}")
            channel_mask_bool = channel_mask.to(device=channel_features.device, dtype=torch.bool)
            key_padding_mask = ~channel_mask_bool  # (N, C), True = ignore

            all_invalid = ~channel_mask_bool.any(dim=1)  # (N,)
            if all_invalid.any():
                num_invalid = int(all_invalid.sum().item())
                logger.warning(
                    "ChannelSetAggregator: %d sample(s) have all channels masked. "
                    "Returning zero embeddings for those samples.",
                    num_invalid,
                )

            valid_rows = ~all_invalid
            if not valid_rows.any():
                return torch.zeros(N, self.output_dim, device=channel_features.device, dtype=channel_features.dtype)

            # Cross-attention on valid rows only (avoid NaNs from all-True key_padding_mask).
            latents_out = torch.zeros(N, self.num_latents, self.output_dim,
                                      device=channel_features.device, dtype=channel_features.dtype)

            x_valid = x[valid_rows]                          # (N', C, D)
            latents_valid = latents[valid_rows]              # (N', L, D)
            kpm_valid = key_padding_mask[valid_rows]         # (N', C)

            query_norm = self.cross_attn_norm(latents_valid)
            attn_out, _ = self.cross_attn(
                query_norm,
                x_valid,
                x_valid,
                key_padding_mask=kpm_valid,
                need_weights=False,
            )
            latents_valid = latents_valid + attn_out
            latents_out[valid_rows] = latents_valid
            latents = latents_out
        else:
            query_norm = self.cross_attn_norm(latents)
            attn_out, _ = self.cross_attn(
                query_norm,
                x,
                x,
                need_weights=False,
            )
            latents = latents + attn_out

        # --- Step 3: Latent self-attention blocks ---
        for latent_layer in self.latent_layers:
            latents = latent_layer(latents)  # (N, L, D)

        # --- Step 4: Flatten and project ---
        flat = latents.reshape(N, -1)  # (N, L*D)
        output = self.output_proj(flat)  # (N, D)

        # Zero out samples that had no valid channels.
        if all_invalid is not None and all_invalid.any():
            output = output.masked_fill(all_invalid.unsqueeze(-1), 0.0)

        return output


class FeatureDistributionCrossAttentionBlock(nn.Module):
    """Cross-attention block where each feature token attends to its support rows."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens_q: torch.Tensor,
        support: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> torch.Tensor:
        if tokens_q.ndim != 3 or support.ndim != 3:
            raise ValueError("tokens_q and support must have shape (N, T, D) and (N, S, D).")
        if support_mask.ndim != 2 or support_mask.shape[:2] != support.shape[:2]:
            raise ValueError(
                f"support_mask must have shape {tuple(support.shape[:2])}, "
                f"got {tuple(support_mask.shape)}."
            )

        support_mask = support_mask.to(device=tokens_q.device, dtype=torch.bool)
        all_invalid = ~support_mask.any(dim=1)

        support_safe = support
        support_mask_safe = support_mask
        if all_invalid.any():
            support_safe = support.clone()
            support_safe[all_invalid] = 0.0
            support_mask_safe = support_mask.clone()
            support_mask_safe[all_invalid, 0] = True

        q = self.norm_q(tokens_q)
        kv = self.norm_kv(support_safe)
        attn_out, _ = self.attn(
            q,
            kv,
            kv,
            key_padding_mask=~support_mask_safe,
            need_weights=False,
        )

        out = tokens_q + self.attn_dropout(attn_out)
        out = out + self.ffn(self.norm_ffn(out))

        if all_invalid.any():
            out = torch.where(all_invalid.view(-1, 1, 1), tokens_q, out)
        return out


class VariableChannelRowWiseFeatureInteractionEncoder(nn.Module):
    """Variable-channel row-wise feature interaction encoder for MantisDA channel features.

    Input / output shape
    --------------------
    x:            (B, T, C, input_dim)
    channel_mask: (B, T, C) bool or None, True = valid channel
    output:       (B, T, output_dim)
    """

    requires_task_context = True

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 512,
        num_groups: int = 8,
        token_dim: int = 128,
        num_cls_tokens: int = 4,
        dist_layers: int = 2,
        dist_heads: int = 4,
        row_layers: int = 3,
        row_heads: int = 4,
        dropout: float = 0.1,
        use_distribution_encoder: bool = True,
        film_alpha_init: float = 0.0,
        use_channel_pos: bool = False,
        max_channels: int = 2048,
        channel_pos_alpha_init: float = 0.0,
        use_group_embedding: bool = True,
        output_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_groups = int(num_groups)
        self.token_dim = int(token_dim)
        self.num_cls_tokens = int(num_cls_tokens)
        self.dist_layers = int(dist_layers)
        self.dist_heads = int(dist_heads)
        self.row_layers = int(row_layers)
        self.row_heads = int(row_heads)
        self.use_distribution_encoder = bool(use_distribution_encoder)
        self.use_channel_pos = bool(use_channel_pos)
        self.max_channels = int(max_channels)
        self.use_group_embedding = bool(use_group_embedding)
        self.output_norm = bool(output_norm)

        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if self.num_groups <= 0:
            raise ValueError("num_groups must be positive.")
        if self.input_dim % self.num_groups != 0:
            raise ValueError(
                f"input_dim ({self.input_dim}) must be divisible by num_groups ({self.num_groups})."
            )
        if self.token_dim <= 0:
            raise ValueError("token_dim must be positive.")
        if self.num_cls_tokens <= 0:
            raise ValueError("num_cls_tokens must be positive.")
        if self.dist_layers < 0:
            raise ValueError("dist_layers must be non-negative.")
        if self.row_layers <= 0:
            raise ValueError("row_layers must be positive.")
        if self.dist_heads <= 0 or self.row_heads <= 0:
            raise ValueError("dist_heads and row_heads must be positive.")
        if self.use_distribution_encoder and self.token_dim % self.dist_heads != 0:
            raise ValueError(
                f"token_dim ({self.token_dim}) must be divisible by dist_heads ({self.dist_heads})."
            )
        if self.token_dim % self.row_heads != 0:
            raise ValueError(
                f"token_dim ({self.token_dim}) must be divisible by row_heads ({self.row_heads})."
            )
        if self.max_channels <= 0:
            raise ValueError("max_channels must be positive.")

        self.group_width = self.input_dim // self.num_groups
        self.group_proj = nn.Sequential(
            nn.LayerNorm(self.group_width),
            nn.Linear(self.group_width, self.token_dim),
        )

        if self.use_group_embedding:
            self.group_embedding = nn.Parameter(torch.zeros(1, 1, 1, self.num_groups, self.token_dim))
        else:
            self.group_embedding = None

        if self.use_channel_pos:
            self.channel_pos_embedding = nn.Embedding(self.max_channels, self.token_dim)
            nn.init.normal_(self.channel_pos_embedding.weight, mean=0.0, std=0.02)
            self.channel_pos_alpha = nn.Parameter(torch.tensor(float(channel_pos_alpha_init)))
        else:
            self.channel_pos_embedding = None
            self.channel_pos_alpha = None

        if self.use_distribution_encoder:
            self.dist_blocks = nn.ModuleList([
                FeatureDistributionCrossAttentionBlock(
                    dim=self.token_dim,
                    num_heads=self.dist_heads,
                    dropout=dropout,
                )
                for _ in range(self.dist_layers)
            ])
            self.film_mlp = nn.Sequential(
                nn.LayerNorm(2 * self.token_dim),
                nn.Linear(2 * self.token_dim, 4 * self.token_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * self.token_dim, 2 * self.token_dim),
            )
            self.film_alpha = nn.Parameter(torch.tensor(float(film_alpha_init)))
        else:
            self.dist_blocks = None
            self.film_mlp = None
            self.film_alpha = None

        self.cls_tokens = nn.Parameter(torch.randn(1, self.num_cls_tokens, self.token_dim) * 0.02)
        row_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=self.row_heads,
            dim_feedforward=4 * self.token_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.row_encoder = nn.TransformerEncoder(row_layer, num_layers=self.row_layers)

        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.num_cls_tokens * self.token_dim),
            nn.Linear(self.num_cls_tokens * self.token_dim, self.output_dim),
        )
        self.out_norm = nn.LayerNorm(self.output_dim) if self.output_norm else nn.Identity()

    def _validate_forward_inputs(
        self,
        x: torch.Tensor,
        train_size: int,
        channel_mask: torch.Tensor | None,
    ) -> tuple[int, int, int, int]:
        if x.ndim != 4:
            raise ValueError("VariableChannelRowWiseFeatureInteractionEncoder expects x with shape (B, T, C, input_dim).")
        B, T, C, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got last dimension {D}.")
        if C <= 0:
            raise ValueError("VariableChannelRowWiseFeatureInteractionEncoder requires at least one channel.")

        S = int(train_size)
        if S <= 0 or S > T:
            raise ValueError(f"train_size must satisfy 0 < train_size <= T ({T}), got {train_size}.")

        if channel_mask is not None:
            if channel_mask.ndim != 3 or channel_mask.shape != (B, T, C):
                raise ValueError(f"channel_mask must have shape ({B}, {T}, {C}), got {tuple(channel_mask.shape)}.")

        if self.use_channel_pos and C > self.max_channels:
            raise ValueError(
                f"VariableChannelRowWiseFeatureInteractionEncoder received {C} channels, "
                f"but max_channels={self.max_channels}."
            )
        return B, T, C, S

    def _apply_distribution_encoder(
        self,
        tokens: torch.Tensor,
        token_mask_flat: torch.Tensor,
        *,
        train_size: int,
    ) -> torch.Tensor:
        B, T, F, d = tokens.shape
        tokens_bftd = tokens.permute(0, 2, 1, 3).contiguous()
        tokens_base = tokens_bftd.reshape(B * F, T, d)
        ctx_q = tokens_base
        support = tokens_base[:, :train_size, :]

        mask_bft = token_mask_flat.permute(0, 2, 1).contiguous()
        support_mask = mask_bft.reshape(B * F, T)[:, :train_size]

        assert self.dist_blocks is not None
        assert self.film_mlp is not None
        assert self.film_alpha is not None
        for block in self.dist_blocks:
            ctx_q = block(ctx_q, support, support_mask)

        film_in = torch.cat([tokens_base, ctx_q], dim=-1)
        gamma, beta = self.film_mlp(film_in).chunk(2, dim=-1)
        film_alpha = self.film_alpha.to(dtype=tokens_base.dtype)
        tokens_film = tokens_base + film_alpha * (torch.tanh(gamma) * tokens_base + beta)

        return tokens_film.reshape(B, F, T, d).permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        *,
        train_size: int,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, C, S = self._validate_forward_inputs(x, train_size, channel_mask)
        G = self.num_groups
        d = self.token_dim

        x_group = x.reshape(B, T, C, G, self.group_width)
        tokens = self.group_proj(x_group)

        if self.group_embedding is not None:
            tokens = tokens + self.group_embedding.to(dtype=tokens.dtype)

        if self.channel_pos_embedding is not None and self.channel_pos_alpha is not None:
            channel_ids = torch.arange(C, device=x.device)
            pos = self.channel_pos_embedding(channel_ids).to(dtype=tokens.dtype).view(1, 1, C, 1, d)
            tokens = tokens + self.channel_pos_alpha.to(dtype=tokens.dtype) * pos

        if channel_mask is None:
            token_mask = torch.ones(B, T, C, G, dtype=torch.bool, device=x.device)
        else:
            channel_mask = channel_mask.to(device=x.device, dtype=torch.bool)
            token_mask = channel_mask.unsqueeze(-1).expand(B, T, C, G)

        F = C * G
        tokens = tokens.reshape(B, T, F, d)
        token_mask_flat = token_mask.reshape(B, T, F)
        tokens = tokens.masked_fill(~token_mask_flat.unsqueeze(-1), 0.0)

        if self.use_distribution_encoder:
            tokens = self._apply_distribution_encoder(
                tokens,
                token_mask_flat,
                train_size=S,
            )
            tokens = tokens.masked_fill(~token_mask_flat.unsqueeze(-1), 0.0)

        row_tokens = tokens.reshape(B * T, F, d)
        row_mask = token_mask_flat.reshape(B * T, F)
        cls = self.cls_tokens.to(dtype=tokens.dtype).expand(B * T, -1, -1)
        row_input = torch.cat([cls, row_tokens], dim=1)

        cls_key_padding_mask = torch.zeros(
            B * T,
            self.num_cls_tokens,
            dtype=torch.bool,
            device=x.device,
        )
        row_key_padding_mask = torch.cat([cls_key_padding_mask, ~row_mask], dim=1)

        row_out = self.row_encoder(row_input, src_key_padding_mask=row_key_padding_mask)
        cls_out = row_out[:, :self.num_cls_tokens, :]
        out = self.output_proj(cls_out.reshape(B * T, self.num_cls_tokens * d))
        out = self.out_norm(out)

        all_invalid_rows = ~row_mask.any(dim=1)
        if all_invalid_rows.any():
            out = out.masked_fill(all_invalid_rows.unsqueeze(-1), 0.0)

        return out.reshape(B, T, self.output_dim)


class TaskAwareChannelAggregator(nn.Module):
    """Task-aware channel aggregator for fixed-sensor multichannel tasks.

    Input / output shape
    --------------------
    channel_features: (B, T, C, P)
    channel_mask:     (B, T, C) bool or None, True = valid channel
    output:           (B, T, D)

    The module keeps the old set-aggregation branch as a residual path, then
    adds a slot-aware branch that conditions each channel on support-set
    statistics before Perceiver-style channel interaction.
    """

    requires_task_context = True

    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 512,
        num_latents: int = 16,
        num_heads: int = 4,
        num_latent_layers: int = 2,
        dropout: float = 0.1,
        max_channels: int = 2048,
        slot_alpha_init: float = 0.0,
        context_alpha_init: float = 0.0,
        slot_gate_init: float = 0.0,
        use_set_residual: bool = True,
        use_residual_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_latents = int(num_latents)
        self.max_channels = int(max_channels)
        self.use_set_residual = bool(use_set_residual)
        self.use_residual_norm = bool(use_residual_norm)
        if self.max_channels <= 0:
            raise ValueError("max_channels must be positive.")

        D = self.output_dim

        # Old set branch. Keep these names aligned with ChannelSetAggregator so
        # old channel_aggregator.* checkpoints can partial-load into this branch.
        self.channel_proj = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, D),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.latents = nn.Parameter(torch.randn(1, num_latents, D) * 0.02)
        self.cross_attn_norm = nn.LayerNorm(D)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=D,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.latent_layers = nn.ModuleList([
            _LatentSelfAttentionBlock(dim=D, num_heads=num_heads, dropout=dropout)
            for _ in range(num_latent_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(num_latents * D),
            nn.Linear(num_latents * D, D),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Slot-aware task-conditioned branch.
        self.channel_id_embedding = nn.Embedding(self.max_channels, D)
        nn.init.normal_(self.channel_id_embedding.weight, mean=0.0, std=0.02)
        self.slot_alpha = nn.Parameter(torch.tensor(float(slot_alpha_init)))
        self.context_alpha = nn.Parameter(torch.tensor(float(context_alpha_init)))
        self.slot_gate = nn.Parameter(torch.tensor(float(slot_gate_init)))

        self.context_mlp = nn.Sequential(
            nn.LayerNorm(2 * D),
            nn.Linear(2 * D, 4 * D),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * D, D),
            nn.Dropout(dropout),
        )

        self.slot_latents = nn.Parameter(torch.randn(1, num_latents, D) * 0.02)
        self.slot_cross_attn_norm = nn.LayerNorm(D)
        self.slot_cross_attn = nn.MultiheadAttention(
            embed_dim=D,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.slot_latent_layers = nn.ModuleList([
            _LatentSelfAttentionBlock(dim=D, num_heads=num_heads, dropout=dropout)
            for _ in range(num_latent_layers)
        ])
        self.slot_output_proj = nn.Sequential(
            nn.LayerNorm(num_latents * D),
            nn.Linear(num_latents * D, D),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_norm = nn.LayerNorm(D) if self.use_residual_norm else nn.Identity()

    def _aggregate_projected_tokens(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None,
        *,
        latents_param: torch.Tensor,
        cross_attn_norm: nn.LayerNorm,
        cross_attn: nn.MultiheadAttention,
        latent_layers: nn.ModuleList,
        output_proj: nn.Sequential,
    ) -> torch.Tensor:
        N, C, D = x.shape
        latents = latents_param.expand(N, -1, -1)
        all_invalid: torch.Tensor | None = None

        if channel_mask is not None:
            if channel_mask.dim() != 2 or channel_mask.shape != (N, C):
                raise ValueError(f"channel_mask must have shape ({N}, {C}), got {tuple(channel_mask.shape)}")
            channel_mask_bool = channel_mask.to(device=x.device, dtype=torch.bool)
            key_padding_mask = ~channel_mask_bool
            all_invalid = ~channel_mask_bool.any(dim=1)
            valid_rows = ~all_invalid
            if not valid_rows.any():
                return torch.zeros(N, D, device=x.device, dtype=x.dtype)

            latents_out = torch.zeros_like(latents)
            query_norm = cross_attn_norm(latents[valid_rows])
            attn_out, _ = cross_attn(
                query_norm,
                x[valid_rows],
                x[valid_rows],
                key_padding_mask=key_padding_mask[valid_rows],
                need_weights=False,
            )
            latents_valid = latents[valid_rows] + attn_out
            latents_out[valid_rows] = latents_valid
            latents = latents_out
        else:
            query_norm = cross_attn_norm(latents)
            attn_out, _ = cross_attn(query_norm, x, x, need_weights=False)
            latents = latents + attn_out

        for latent_layer in latent_layers:
            latents = latent_layer(latents)

        output = output_proj(latents.reshape(N, -1))
        if all_invalid is not None and all_invalid.any():
            output = output.masked_fill(all_invalid.unsqueeze(-1), 0.0)
        return output

    def _support_context(
        self,
        x: torch.Tensor,
        train_size: int,
        channel_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B, T, C, D = x.shape
        S = int(train_size)
        if S <= 0:
            raise ValueError("train_size must be positive for TaskAwareChannelAggregator.")
        S = min(S, T)
        support = x[:, :S]  # (B, S, C, D)

        if channel_mask is None:
            mu = support.mean(dim=1)
            std = support.std(dim=1, unbiased=False)
        else:
            support_mask = channel_mask[:, :S].to(device=x.device, dtype=torch.bool)
            weights = support_mask.unsqueeze(-1).to(dtype=x.dtype)
            denom = weights.sum(dim=1).clamp_min(1.0)  # (B, C, 1)
            mu = (support * weights).sum(dim=1) / denom
            centered = (support - mu.unsqueeze(1)) * weights
            var = centered.square().sum(dim=1) / denom
            std = torch.sqrt(var.clamp_min(0.0) + 1e-6)
            no_support = ~support_mask.any(dim=1)
            if no_support.any():
                mu = mu.masked_fill(no_support.unsqueeze(-1), 0.0)
                std = std.masked_fill(no_support.unsqueeze(-1), 0.0)

        return self.context_mlp(torch.cat([mu, std], dim=-1))  # (B, C, D)

    def forward(
        self,
        channel_features: torch.Tensor,
        train_size: int,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if channel_features.ndim != 4:
            raise ValueError("TaskAwareChannelAggregator expects channel_features with shape (B, T, C, P)")
        B, T, C, _P = channel_features.shape
        if C > self.max_channels:
            raise ValueError(
                f"TaskAwareChannelAggregator received {C} channels, "
                f"but max_channels={self.max_channels}."
            )
        if channel_mask is not None:
            if channel_mask.dim() != 3 or channel_mask.shape != (B, T, C):
                raise ValueError(f"channel_mask must have shape ({B}, {T}, {C}), got {tuple(channel_mask.shape)}")
            channel_mask = channel_mask.to(device=channel_features.device, dtype=torch.bool)

        flat_mask = None if channel_mask is None else channel_mask.reshape(B * T, C)
        x = self.channel_proj(channel_features.reshape(B * T, C, -1)).reshape(B, T, C, self.output_dim)

        z_set = self._aggregate_projected_tokens(
            x.reshape(B * T, C, self.output_dim),
            flat_mask,
            latents_param=self.latents,
            cross_attn_norm=self.cross_attn_norm,
            cross_attn=self.cross_attn,
            latent_layers=self.latent_layers,
            output_proj=self.output_proj,
        ).reshape(B, T, self.output_dim)

        channel_ids = torch.arange(C, device=channel_features.device)
        slot_code = self.channel_id_embedding(channel_ids).to(dtype=x.dtype)
        x_slot = x + self.slot_alpha.to(dtype=x.dtype) * slot_code.view(1, 1, C, self.output_dim)

        context = self._support_context(x_slot, train_size=int(train_size), channel_mask=channel_mask)
        x_ctx = x_slot + self.context_alpha.to(dtype=x.dtype) * context.unsqueeze(1)

        z_slot = self._aggregate_projected_tokens(
            x_ctx.reshape(B * T, C, self.output_dim),
            flat_mask,
            latents_param=self.slot_latents,
            cross_attn_norm=self.slot_cross_attn_norm,
            cross_attn=self.slot_cross_attn,
            latent_layers=self.slot_latent_layers,
            output_proj=self.slot_output_proj,
        ).reshape(B, T, self.output_dim)

        if self.use_set_residual:
            return self.residual_norm(z_set + self.slot_gate.to(dtype=z_set.dtype) * z_slot)
        return z_slot


class TaskAwareSlotOnlyChannelAggregator(nn.Module):
    """Task-aware slot-only channel aggregator.

    Input / output shape
    --------------------
    channel_features: (B, T, C, P)
    channel_mask:     (B, T, C) bool or None, True = valid channel
    output:           (B, T, D)

    This variant keeps only the task-conditioned slot branch. It intentionally
    does not define the old set-aggregation branch or a slot gate.
    """

    requires_task_context = True

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 512,
        num_latents: int = 16,
        num_heads: int = 4,
        num_latent_layers: int = 2,
        dropout: float = 0.1,
        max_channels: int = 2048,
        slot_alpha_init: float = 0.0,
        context_alpha_init: float = 0.0,
        use_residual_norm: bool = True,
        slot_gate_init: float | None = None,
        **_unused_kwargs,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_latents = int(num_latents)
        self.max_channels = int(max_channels)
        self.use_residual_norm = bool(use_residual_norm)
        if self.max_channels <= 0:
            raise ValueError("max_channels must be positive.")

        D = self.output_dim

        self.channel_proj = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, D),
        )

        self.channel_id_embedding = nn.Embedding(self.max_channels, D)
        nn.init.normal_(self.channel_id_embedding.weight, mean=0.0, std=0.02)
        self.slot_alpha = nn.Parameter(torch.tensor(float(slot_alpha_init)))
        self.context_alpha = nn.Parameter(torch.tensor(float(context_alpha_init)))

        self.context_mlp = nn.Sequential(
            nn.LayerNorm(D * 2),
            nn.Linear(D * 2, D * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D * 4, D),
            nn.Dropout(dropout),
        )

        self.slot_latents = nn.Parameter(torch.randn(1, num_latents, D) * 0.02)
        self.slot_cross_attn_norm = nn.LayerNorm(D)
        self.slot_kv_norm = nn.LayerNorm(D)
        self.slot_cross_attn = nn.MultiheadAttention(
            embed_dim=D,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.slot_latent_layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(D),
                "attn": nn.MultiheadAttention(
                    embed_dim=D,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                ),
                "norm2": nn.LayerNorm(D),
                "mlp": nn.Sequential(
                    nn.Linear(D, D * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(D * 4, D),
                    nn.Dropout(dropout),
                ),
            })
            for _ in range(num_latent_layers)
        ])

        self.slot_output_proj = nn.Sequential(
            nn.LayerNorm(num_latents * D),
            nn.Linear(num_latents * D, D),
        )
        self.output_norm = nn.LayerNorm(D) if self.use_residual_norm else nn.Identity()

    def _support_context(
        self,
        x: torch.Tensor,
        train_size: int,
        channel_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B, _T, C, D = x.shape
        support = x[:, :train_size]  # (B, S, C, D)

        if channel_mask is None:
            mu = support.mean(dim=1)
            std = support.std(dim=1, unbiased=False)
        else:
            support_mask = channel_mask[:, :train_size].to(device=x.device, dtype=torch.bool)
            weights = support_mask.unsqueeze(-1).to(dtype=x.dtype)
            denom_raw = weights.sum(dim=1)  # (B, C, 1)
            valid = denom_raw > 0
            denom = denom_raw.clamp_min(1.0)

            mu = (support * weights).sum(dim=1) / denom
            var = (((support - mu.unsqueeze(1)) ** 2) * weights).sum(dim=1) / denom
            std = torch.sqrt(var.clamp_min(1e-6))

            zeros = torch.zeros(B, C, D, device=x.device, dtype=x.dtype)
            mu = torch.where(valid, mu, zeros)
            std = torch.where(valid, std, zeros)

        return self.context_mlp(torch.cat([mu, std], dim=-1))

    def forward(
        self,
        x: torch.Tensor,
        *,
        train_size: int,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("TaskAwareSlotOnlyChannelAggregator expects x with shape (B, T, C, P)")
        B, T, C, _P = x.shape
        if C <= 0:
            raise ValueError("TaskAwareSlotOnlyChannelAggregator requires at least one channel.")
        if int(train_size) <= 0 or int(train_size) > T:
            raise ValueError(f"train_size must be in [1, {T}], got {train_size}.")
        if C > self.max_channels:
            raise ValueError(
                f"TaskAwareSlotOnlyChannelAggregator received {C} channels, "
                f"but max_channels={self.max_channels}."
            )
        if channel_mask is not None:
            if channel_mask.dim() != 3 or channel_mask.shape != (B, T, C):
                raise ValueError(f"channel_mask must have shape ({B}, {T}, {C}), got {tuple(channel_mask.shape)}")
            channel_mask = channel_mask.to(device=x.device, dtype=torch.bool)

        x = self.channel_proj(x)  # (B, T, C, D)

        channel_ids = torch.arange(C, device=x.device)
        channel_embed = self.channel_id_embedding(channel_ids).to(dtype=x.dtype)
        x_slot = x + self.slot_alpha.to(dtype=x.dtype) * channel_embed.view(1, 1, C, self.output_dim)

        context = self._support_context(x_slot, train_size=int(train_size), channel_mask=channel_mask)
        x_ctx = x_slot + self.context_alpha.to(dtype=x.dtype) * context.unsqueeze(1)

        N = B * T
        x_flat = x_ctx.reshape(N, C, self.output_dim)
        latents = self.slot_latents.to(dtype=x.dtype).expand(N, -1, -1)

        key_padding_mask = None
        all_invalid = None
        if channel_mask is not None:
            mask_flat = channel_mask.reshape(N, C)
            all_invalid = ~mask_flat.any(dim=1)
            key_padding_mask = ~mask_flat
            if all_invalid.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_invalid, 0] = False

        q = self.slot_cross_attn_norm(latents)
        kv = self.slot_kv_norm(x_flat)
        attn_out, _ = self.slot_cross_attn(
            q,
            kv,
            kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        latents = latents + attn_out

        for layer in self.slot_latent_layers:
            h = layer["norm1"](latents)
            attn_out, _ = layer["attn"](h, h, h, need_weights=False)
            latents = latents + attn_out
            latents = latents + layer["mlp"](layer["norm2"](latents))

        out = self.slot_output_proj(latents.reshape(N, self.num_latents * self.output_dim))
        out = out.reshape(B, T, self.output_dim)
        out = self.output_norm(out)

        if all_invalid is not None and all_invalid.any():
            out = out.masked_fill(all_invalid.reshape(B, T, 1), 0.0)

        return out
