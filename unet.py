import math

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from Model.CBAM import CBAM

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class SpatialTransformer(nn.Module):
    """
    ## Spatial Transformer
    """

    def __init__(self, channels: int, n_heads: int, n_layers: int):
        """
        :param channels: is the number of channels in the feature map
        :param n_heads: is the number of attention heads
        :param n_layers: is the number of transformer layers
        :param d_cond: is the size of the conditional embedding
        """
        super(SpatialTransformer, self).__init__()
        # Initial group normalization
        self.norm = torch.nn.GroupNorm(num_groups=16, num_channels=channels, eps=1e-6, affine=True)
        # Initial $1 \times 1$ convolution
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(channels, n_heads, 32) for _ in range(n_layers)]
        )

        # Final $1 \times 1$ convolution
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the feature map of shape `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings of shape `[batch_size,  n_cond, d_cond]`
        """
        # Get shape `[batch_size, channels, height, width]`
        b, c, h, w = x.shape
        # For residual connection
        x_in = x
        # Normalize
        x = self.norm(x)
        # Initial $1 \times 1$ convolution
        x = self.proj_in(x)
        # Transpose and reshape from `[batch_size, channels, height, width]`
        # to `[batch_size, height * width, channels]`
        x = x.permute(0, 2, 3, 1).view(b, h * w, c)
        # Apply the transformer layers
        for block in self.transformer_blocks:
            x = block(x)
        # Reshape and transpose from `[batch_size, height * width, channels]`
        # to `[batch_size, channels, height, width]`
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        # Final $1 \times 1$ convolution
        x = self.proj_out(x)
        # Add residual
        return x + x_in


class BasicTransformerBlock(nn.Module):
    """
    ### Transformer Layer
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        """
        super(BasicTransformerBlock, self).__init__()
        # Self-attention layer and pre-norm layer
        self.attn1 = CrossAttention(d_model, d_model, n_heads, d_head)
        self.norm1 = nn.LayerNorm(d_model)
        # Feed-forward network and pre-norm layer
        self.ff = FeedForward(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        """
        :param x: are the input embeddings of shape `[batch_size, height * width, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size,  n_cond, d_cond]`
        """
        # Self attention
        x = self.attn1(self.norm1(x)) + x

        # Feed-forward network
        x = self.ff(self.norm3(x)) + x
        #
        return x


class CrossAttention(nn.Module):
    """
    ### Cross Attention Layer

    This falls-back to self-attention when conditional embeddings are not specified.
    """

    use_flash_attention: bool = False

    def __init__(self, d_model: int, d_cond: int, n_heads: int, d_head: int):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        :param d_cond: is the size of the conditional embeddings
        :param is_inplace: specifies whether to perform the attention softmax computation inplace to
            save memory
        """
        super(CrossAttention, self).__init__()

        self.n_heads = n_heads
        self.d_head = d_head

        # Attention scaling factor
        self.scale = d_head ** -0.5

        # Query, key and value mappings
        d_attn = d_head * n_heads
        self.to_q = nn.Linear(d_model, d_attn, bias=False)
        self.to_k = nn.Linear(d_cond, d_attn, bias=False)
        self.to_v = nn.Linear(d_cond, d_attn, bias=False)

        # Final linear layer
        self.to_out = nn.Sequential(nn.Linear(d_attn, d_model))

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """
        :param x: are the input embeddings of shape `[batch_size, height * width, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size, n_cond, d_cond]`
        """

        # If `cond` is `None` we perform self attention
        has_cond = cond is not None
        if not has_cond:
            cond = x

        # Get query, key and value vectors
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)

        return self.normal_attention(q, k, v)

    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Normal Attention

        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """

        # Split them to heads of shape `[batch_size, seq_len, n_heads, d_head]`
        q = q.view(*q.shape[:2], self.n_heads, -1)  # preserved  first two dimensions of the tensor
        k = k.view(*k.shape[:2], self.n_heads, -1)
        v = v.view(*v.shape[:2], self.n_heads, -1)

        # Calculate attention $\frac{Q K^\top}{\sqrt{d_{key}}}$
        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale

        # Compute softmax
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)$$
        attn = attn.softmax(dim=-1)

        # Compute attention output
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        out = torch.einsum('bhij,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, height * width, n_heads * d_head]`
        out = out.reshape(*out.shape[:2], -1)
        # Map to `[batch_size, height * width, d_model]` with a linear layer
        return self.to_out(out)


class FeedForward(nn.Module):
    """
    ### Feed-Forward Network
    """

    def __init__(self, d_model: int, d_mult: int = 4):
        """
        :param d_model: is the input embedding size
        :param d_mult: is multiplicative factor for the hidden layer size
        """
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            GeGLU(d_model, d_model * d_mult),
            nn.Dropout(0),
            nn.Linear(d_model * d_mult, d_model)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class GeGLU(nn.Module):
    """
    ### GeGLU Activation

    $$\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$$
    """

    def __init__(self, d_in: int, d_out: int):
        super(GeGLU, self).__init__()
        # Combined linear projections $xW + b$ and $xV + c$
        self.proj = nn.Linear(d_in, d_out * 2)

    def forward(self, x: torch.Tensor):
        # Get $xW + b$ and $xV + c$
        x, gate = self.proj(x).chunk(2, dim=-1)  # splits the output tensor into two parts along the last dimension
        # $\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$
        return x * F.gelu(gate)


# =========================================PE2D=========================================================================
class PositionalEncoding2D(nn.Module):
    def __init__(self):
        super(PositionalEncoding2D, self).__init__()

    def forward(self, x):
        # input tensor shape: [b, c, h, w]
        b, c, h, w = x.shape
        pe = self.positionalencoding2d(c, h, w).unsqueeze(0).repeat(b, 1, 1, 1).to(x.device)
        return x + pe

    def positionalencoding2d(self, d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe


# =========================================UNET=========================================================================


class DownSample(nn.Module):
    """
    ## Down-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super(DownSample, self).__init__()
        # $3 \times 3$ convolution with stride length of $2$ to down-sample by a factor of $2$
        # self.op = nn.Conv2d(channels, channels, kernel_size=2, stride=2, padding=1)
        self.op = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Apply convolution
        return self.op(x)


class DoubleConv(nn.Module):
    """
    ## ResNet Block
    """

    def __init__(self, channels: int, out_channels=None):
        """
        :param channels: the number of input channels

        :param out_channels: is the number of out channels. defaults to `channels.
        """
        super(DoubleConv, self).__init__()
        # `out_channels` not specified
        if out_channels is None:
            out_channels = channels

        # First normalization and convolution
        self.in_layers = nn.Sequential(
            # normalization(channels),

            nn.Conv2d(channels, out_channels, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
            # CBAM(out_channels, reduction_ratio=4),
        )

        self.out_layers = nn.Sequential(
            # normalization(out_channels),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
            # CBAM(out_channels, reduction_ratio=8),
        )

        # `channels` to `out_channels` mapping layer for residual connection
        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Initial convolution
        h = self.in_layers(x)
        # Final convolution
        h = self.out_layers(h)
        # Add skip connection
        # return self.skip_connection(x) + h
        return h


class InceptionResNetA(nn.Module):
    def __init__(self, in_channels, out_channels, final_out_channels):
        super(InceptionResNetA, self).__init__()
        self.branch_0 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(out_channels, int(out_channels * 1.5), 3, stride=1, padding=1, bias=False),
            nn.Conv2d(int(out_channels * 1.5), int(out_channels * 2), 3, stride=1, padding=1, bias=False),
        )
        self.conv = nn.Conv2d(out_channels * 4, final_out_channels, 1, stride=1, padding=0, bias=True)
        self.main = nn.Conv2d(in_channels, final_out_channels, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU()
        self.bn1 = normalization(out_channels)
        self.bn2 = normalization(out_channels)
        self.bn3 = normalization(int(out_channels * 2))
        self.attention_block = CBAM(final_out_channels, reduction_ratio=8, no_spatial=False)

    def forward(self, x):
        x0 = self.branch_0(x)
        x0 = self.bn1(x0)
        x1 = self.branch_1(x)
        x1 = self.bn2(x1)
        x2 = self.branch_2(x)
        x2 = self.bn3(x2)
        x_res = torch.cat((x0, x1, x2), dim=1)
        x_res = self.conv(x_res)
        x_main = self.main(x)
        out = self.relu(x_main + x_res)
        final_out = self.attention_block(out) + out
        # final_out = out    # no CBAM
        return final_out


class GroupNorm16(nn.GroupNorm):
    """
    ### Group normalization with float32 casting
    """

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    ### Group normalization

    This is a helper function, with fixed number of groups..
    """
    return GroupNorm16(16, channels)


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels, features=[16, 32, 64, 128], n_head=4, n_layer=2):
        super(UNET, self).__init__()
        self.input = nn.Conv2d(in_channels, features[0], 3, padding=1)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        self.pool = nn.ModuleList()
        for feature in features:
            self.pool.append(DownSample(feature))

        # Down part of UNET
        in_channels = features[0]
        for feature in features:
            # self.downs.append(DoubleConv(in_channels, feature))
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(in_channels=feature*2, out_channels=feature, kernel_size=2, stride=2)
            )
            # self.ups.append(DoubleConv(feature*2, feature))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottom = nn.Sequential(nn.Conv2d(features[-1], features[-1] * 2, kernel_size=1, padding=0, bias=False),
                                    # DoubleConv(features[-1], features[-1] * 2),
                                    PositionalEncoding2D(),
                                    SpatialTransformer(channels=features[-1] * 2, n_heads=n_head, n_layers=n_layer),
                                    )
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        # input tensor shape: [b, c, 50, 50]
        x = self.input(x)
        skip_connections = []

        # down sampling
        for i, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            x = self.pool[i](x)

        x = self.bottom(x)
        skip_connections = skip_connections[::-1]    # reverse list

        # up sampling
        # notice: we do up + DoubleConv per step
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # check if the two cat tensors match during skip connection
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        out = self.final_conv(x)
        # sat_mask = out < 0
        # out[sat_mask] = 0.001
        return out


if __name__ == "__main__":
    my_device = 'cuda'

    x1 = torch.randn((4, 5, 50, 50)).to(my_device)
    print(f'x has shape: {x1.shape}')

    model = UNET(5, 1).to(my_device)

    pred = model(x1)

    print(f'pred has shape: {pred.shape}')  # out: (16, 84, 6, 50, 50)
    print(f'total params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
