
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)  # Shape: (B, embed_dim, H//patch_size, W//patch_size)
class PatchMerging(nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.reduction = nn.Linear(4*input_dim, output_dim, bias=False)
    self.activation = nn.GELU()

  def forward(self, x):
    B, C, H, W = x.shape
    x = x.view(B, C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(B, H // 2, W // 2, 4 * C)
    x = self.reduction(x)
    x = self.activation(x)
    return x.permute(0, 3, 1, 2).contiguous()

class WindowAttention(nn.Module):
  def __init__(self, embed_dim, num_heads, window_size):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads,
    self.window_size = window_size
    self.attention = nn.MultiheadAttention(embed_dim, num_heads)
    self.norm1 = nn.LayerNorm(embed_dim)

  def forward(self, x):

    B, C, H, W = x.shape

    x = x.view(B , C, H//self.window_size, W//self.window_size, self.window_size, self.window_size)

    x = x.permute(0, 2, 3, 4, 5, 1)

    x = x.view(B * (H//self.window_size * W//self.window_size), self.window_size*self.window_size, C)


    x = x.permute(1, 0, 2)

    attn_score , _ = self.attention(x, x, x)
    attn_score = self.norm1(attn_score)
    attn_score = attn_score.permute(0, 1, 2).contiguous()

    x = attn_score.view(B, C, H, W)

    return x

class S_MSA(nn.Module):
  def __init__(self, embed_dim, num_heads, window_size, shift_size=0):
    super().__init__()
    self.shift_size = shift_size
    self.attention = WindowAttention(embed_dim, num_heads, window_size)

  def forward(self, x):
    B, C, H, W = x.shape
    if self.shift_size > 0:
      x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
      x = self.attention(x)
      x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
    return x


class SwinStage(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, window_size=7):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(embed_dim, num_heads, window_size=window_size, shift_size=(i % 2) * window_size // 2)
            for i in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class SwinTransformerTiny(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_channels=3, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2]):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)

        # Stage definitions with Patch Merging between stages
        self.stage1 = SwinStage(embed_dim=embed_dim, num_heads=3, depth=depths[0])
        self.merge1 = PatchMerging(input_dim=embed_dim, output_dim=embed_dim * 2)

        self.stage2 = SwinStage(embed_dim=embed_dim * 2, num_heads=6, depth=depths[1])
        self.merge2 = PatchMerging(input_dim=embed_dim * 2, output_dim=embed_dim * 4)

        self.stage3 = SwinStage(embed_dim=embed_dim * 4, num_heads=12, depth=depths[2])
        self.merge3 = PatchMerging(input_dim=embed_dim * 4, output_dim=embed_dim * 8)

        self.stage4 = SwinStage(embed_dim=embed_dim * 8, num_heads=24, depth=depths[3])

        # Classification head
        self.head = nn.Linear(embed_dim * 8, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)

        x = self.stage1(x)
        x = self.merge1(x)

        x = self.stage2(x)
        x = self.merge2(x)

        x = self.stage3(x)
        x = self.merge3(x)

        x = self.stage4(x)

        # Global average pooling
        x = x.mean(dim=[-2, -1])  # Shape: (B, embed_dim * 8)

        # Classification head
        x = self.head(x)
        return x
