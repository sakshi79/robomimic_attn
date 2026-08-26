import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class SiglipVisionConfig:
  num_channels: int = 3
  embed_dim: int = 768
  image_size: int = 224
  patch_size: int = 16

class SiglipVisionEmbeddings(nn.Module):
  def __init__(self, config: SiglipVisionConfig):
    super().__init__()
    self.config = config

    self.num_channels = config.num_channels
    self.embed_dim = config.embed_dim
    self.image_size = config.image_size
    self.patch_size = config.patch_size

    self.patch_embedding = nn.Conv2d(
        in_channels = self.num_channels,
        out_channels = self.embed_dim,
        kernel_size = self.patch_size,
        stride = self.patch_size,
        padding = "valid",
    )

    self.num_patches = (self.image_size//self.patch_size)**2
    self.num_positions = self.num_patches
    self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
    self.register_buffer(
        "position_ids",
        torch.arange(self.num_positions).expand((1,-1)),
        persistent = False,
    )

  def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
    B,C,H,W = pixel_values.shape
    patch_embeds = self.patch_embedding(pixel_values)
    embeddings = patch_embeds.flatten(start_dim=2, end_dim=-1)
    embeddings = embeddings.transpose(1,2)
    embeddings = embeddings + self.position_embedding(self.position_ids)
    return embeddings


class Head(nn.Module):
  """ A single head of the multi-head self-attention module """

  def __init__(self, n_in, n_head, context_length):
    super().__init__()
    self.head_size = n_head
    self.key = nn.Linear(n_in, n_head, bias=False)
    self.query = nn.Linear(n_in, n_head, bias=False)
    self.value = nn.Linear(n_in, n_head, bias=False)

  def forward(self,x):
    B,T,C = x.shape   # Get batch size, number of tokens, and embedding size
    k = self.key(x)   # key: information contained in the token
    q = self.query(x) # query: information the token is looking for
    v = self.value(x) # value: aggregated if there's a match
    wei = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(self.head_size)) # Normalized attnetion between query and key
    wei = F.softmax(wei, dim=-1)  # Raw attention scores to attention probability distribution
    out = wei @ v  # use attention weights to aggregate values
    return out

class MultiHeadAttention(nn.Module):
  """ Multi-head attention module
  Run all heads in parallel, then concatenate their output """

  def __init__(self, num_head, n_in, head_size, context_length):
    super().__init__()
    self.head_size = head_size
    self.num_head = num_head
    # The more heads we have, the more interesting features and relationships we can capture
    self.heads = [Head(n_in, head_size, context_length) for _ in range(num_head)]  # Each head can focus on different specific features, eg colors, textures
    self.proj = nn.Linear(n_in, n_in)  # Project output to the same dimension as input

  def forward(self,x):
    out = [h(x) for h in self.heads]
    out = torch.concat(out, -1)
    out = self.proj(out)
    return out


@dataclass
class SiglipVisionConfig:
  num_channels: int = 3
  image_size: int = 224
  patch_size: int = 16
  num_attention_heads: int = 12 # The more heads we have, the more interesting features and relationships we can capture.
  # Each head can focus on different specific features, eg colors, textures
  hidden_size: int = 768
  attention_dropout: int = 0.0

class SiglipAttention(nn.Module):
  def __init__(self, config: SiglipVisionConfig):
    super().__init__()
    self.config = config
    self.embed_dim = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.dropout = config.attention_dropout

    self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
    self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
    self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
    self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

  def forward(self, hidden_states):
    # hidden states are the embeddings of the patches
    B,T,C = hidden_states.shape  # batch size, num_patches, embedding_size

    k_states = self.k_proj(hidden_states)
    q_states = self.q_proj(hidden_states)
    v_states = self.v_proj(hidden_states)

    # To process heads in parallel
    k_states = k_states.view(B, T, self.num_heads, C//self.num_heads).transpose(1,2)
    q_states = q_states.view(B, T, self.num_heads, C//self.num_heads).transpose(1,2)
    v_states = v_states.view(B, T, self.num_heads, C//self.num_heads).transpose(1,2)

    attn_weights = (q_states @ k_states.transpose(-2,-1)) * (1.0 / math.sqrt(k_states.size(-1)))
    attn_weights = F.softmax(attn_weights, dim=-1).to(q_states.dtype)
    attn_weights = F.dropout(attn_weights, p=self.dropout, training = self.training)
    attn_outs = attn_weights @ v_states
    attn_outs = attn_outs.transpose(1,2)
    attn_outs = attn_outs.reshape(B,T,C).contiguous()  # Ensuring the tensors are contiguous in memory allocation for efficient memory consumption
    attn_outs = self.out_proj(attn_outs)
    return attn_outs


# batch_size = 1
# num_patches = 196
# embed_dim = 768

# hidden_states = torch.randn(batch_size, num_patches, embed_dim)
# config = SiglipVisionConfig(
#     attention_dropout = 0.0,
#     num_attention_heads = 12,
#     hidden_size = 768
# )

# attention = SiglipAttention(config)
# output = attention(hidden_states)

# print(f"Input shape: {hidden_states.shape}")
# print(f"Output shape: {output.shape}")
# # They must have the same shape!

@dataclass
class SiglipVisionConfig:
  num_channels: int = 3
  image_size: int = 224
  patch_size: int = 16
  num_attention_heads: int = 12
  hidden_size: int = 768
  attention_dropout: float = 0.0
  intermediate_size: int = 3072

class SiglipMLP(nn.Module):
  def __init__(self, config: SiglipVisionConfig):
    super().__init__()
    self.config = config
    self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
    # By mapping the parameters to higher dimensions, the model can now learn more complex relations.
    self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.fc1(hidden_states)
    hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')
    hidden_states = self.fc2(hidden_states)
    return hidden_states

# mlp = SiglipMLP(SiglipVisionConfig(hidden_size=768, intermediate_size=3072))
# mlp(torch.randn([1,196,768])).shape

@dataclass
class SiglipVisionConfig:
  num_channels: int = 3
  image_size: int = 224
  patch_size: int = 16
  num_attention_heads: int = 12
  hidden_size: int = 768
  attention_dropout: float = 0.0
  intermediate_size: int = 3072
  layer_norm_eps: float = 1e-6

class SiglipVisionEmbeddings(nn.Module):
  def __init__(self, config: SiglipVisionConfig):
    super().__init__()
    self.config = config

    self.num_channels = config.num_channels
    self.image_size = config.image_size
    self.embed_dim = config.hidden_size
    self.patch_size = config.patch_size

    self.patch_embedding = nn.Conv2d(
        in_channels = self.num_channels,
        out_channels = self.embed_dim,
        kernel_size = self.patch_size,
        stride = self.patch_size,
        padding='valid',
    )

    self.num_patches = (self.image_size // self.patch_size) ** 2
    self.num_positions = self.num_patches
    self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim) # Create embedding of size embed_dim for every i in range(num_positions)
    self.register_buffer(
        "position_ids",
        torch.arange(self.num_positions).expand((1,-1)),
        persistent = False,
    )

  def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
    B,C,H,W = pixel_values.shape
    patch_embeds = self.patch_embedding(pixel_values)
    embeddings = patch_embeds.flatten(start_dim = 2, end_dim = -1)
    embeddings = embeddings.transpose(1,2)
    embeddings = embeddings + self.position_embedding(self.position_ids)
    return embeddings


class SiglipEncoderLayer(nn.Module):
  def __init__(self, config = SiglipVisionConfig):
    super().__init__()
    self.embed_dim = config.hidden_size
    self.self_attn = SiglipAttention(config)
    self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)
    self.mlp = SiglipMLP(config)
    self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)

  def forward(self, hidden_states):
    residual = hidden_states
    hidden_states = self.layer_norm1(hidden_states)
    hidden_states = self.self_attn(hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states

# encoder_layer = SiglipEncoderLayer(SiglipVisionConfig(hidden_size=768, intermediate_size=3072))
# encoder_layer(torch.randn(1,196,768)).shape

@dataclass
class SiglipVisionConfig:
  num_hidden_layers: int = 12  # same as original layer
  num_channels: int = 3
  image_size: int = 224
  patch_size: int = 16
  num_attention_heads: int = 12
  hidden_size: int = 768
  intermediate_size: int = 3072
  layer_norm_eps: float = 1e-6
  attention_dropout: float = 0.0

class SiglipEncoder(nn.Module):
  def __init__(self, config: SiglipVisionConfig):
    super().__init__()
    self.config = config
    self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

  def forward(self, hidden_states):
    for layer in self.layers:
      hidden_states = layer(hidden_states)
    return hidden_states


# encoder = SiglipEncoder(SiglipVisionConfig(hidden_size=768, intermediate_size=3072))
# encoder(torch.randn(1, 196, 768)).shape

class SiglipVisionTransformer(nn.Module):
  def __init__(self, config: SiglipVisionConfig):
    super().__init__()
    self.config = config
    self.embeddings = SiglipVisionEmbeddings(config)
    self.encoder = SiglipEncoder(config)
    self.post_layernorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)

  def forward(self, pixel_values):
    hidden_states = self.embeddings(pixel_values)
    last_hidden_state = self.encoder(hidden_states)
    last_hidden_state = self.post_layernorm(last_hidden_state)
    return last_hidden_state

# siglip = SiglipVisionTransformer(SiglipVisionConfig(hidden_size=768, intermediate_size = 3072))
# siglip(image_tensor).shape

class SiglipVisionModel(nn.Module):
  def __init__(self, config: SiglipVisionConfig):
    super().__init__()
    self.config = config
    self.vision_model = SiglipVisionTransformer(config)

  def forward(self, pixel_values):
    return self.vision_model(pixel_values)

# siglip = SiglipVisionModel(SiglipVisionConfig(hidden_size = 768, intermediate_size = 3072))
# siglip(image_tensor).shape

class SigLIPMasker(nn.Module):
    """
    Spatial attention mask using SigLIP vision encoder.

    Works like SoftAttentionMask from mask.py:
      Input:  (B, C, H, W) image tensor
      Output: (B, C, H, W) attention-masked image tensor

    Internally, SigLIP produces per-patch embeddings which are projected
    to scalar attention scores, reshaped to a 2D spatial grid, upsampled
    to the original image resolution, and broadcast-multiplied with the
    input image.
    """

    def __init__(self, config=None, in_channels=3):
        super().__init__()
        if config is None:
            config = SiglipVisionConfig(hidden_size=768, intermediate_size=3072)
        self.config = config
        self.vision = SiglipVisionModel(config)

        # Project each patch embedding → scalar attention score
        self.attn_head = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        # Spatial grid size (patches per side)
        self.grid_size = config.image_size // config.patch_size

    def forward(self, pixel_values):
        B, C, H, W = pixel_values.shape

        # SigLIP expects a fixed input size; resize if needed
        siglip_input = pixel_values
        if H != self.config.image_size or W != self.config.image_size:
            siglip_input = F.interpolate(
                pixel_values,
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            )

        # (B, num_patches, hidden_size)
        patch_embeddings = self.vision(siglip_input)

        # Per-patch attention scores → (B, num_patches, 1)
        attn_scores = self.attn_head(patch_embeddings)

        # Reshape to spatial grid → (B, 1, grid_h, grid_w)
        attn_map = attn_scores.view(B, 1, self.grid_size, self.grid_size)

        # Sigmoid to get [0, 1] attention mask
        attn_map = torch.sigmoid(attn_map)

        # Upsample to original image resolution
        if attn_map.shape[-2:] != (H, W):
            attn_map = F.interpolate(
                attn_map,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        # Element-wise multiply: same contract as SoftAttentionMask
        return attn_map * pixel_values