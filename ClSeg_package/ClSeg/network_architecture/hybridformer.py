import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class Attention(nn.Module):
    def __init__(self, dim, window_size=3):
        super().__init__()
        # self.conv1 = nn.Conv3d(dim, dim, 1)
        self.pool = nn.AvgPool3d(kernel_size=window_size, stride=1, padding=window_size // 2)
        # self.conv2 = nn.Conv3d(dim, dim, 1)
        # self.act_layer = nn.ReLU()

    def forward(self, x):
        # x = self.act_layer(self.conv1(x))
        x = self.pool(x)
        # x = self.conv2(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, window_size=3,
                 mlp_ratio=4., drop=0., drop_path=0., norm_layer=LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim, data_format="channels_first")
        self.norm2 = norm_layer(dim, data_format="channels_last")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.token_mixer = Attention(dim, window_size)

        self.conv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim),
            nn.BatchNorm3d(dim),
            nn.ReLU(),
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.conv(x)
        x = einops.rearrange(x, 'b c d h w -> b (d h w) c', b=B, c=C, d=D, h=H, w=W)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = einops.rearrange(x, 'b (d h w) c -> b c d h w', b=B, c=C, d=D, h=H, w=W)
        return x


class BasicLayer(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 scale_factor=None,
                 out_dim=None,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(indim=dim, outdim=out_dim, scale_factor=scale_factor)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        return x, x

class Subsample(nn.Module):
    def __init__(self, indim, outdim, scale_factor=(1, 1, 1), mode='trilinear'):
        super(Subsample, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.scale_factor = scale_factor
        self.mode = mode
        self.norm = nn.InstanceNorm3d(outdim)
        self.lrelu = nn.LeakyReLU()
        self.subsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.proj = nn.Conv3d(self.indim, self.outdim, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.subsample(x)
        x = self.proj(x)
        return x


class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """

    def __init__(self, patch_size=(2, 2, 2), in_chans=1, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        B, C, D, H, W = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).view(-1, self.embed_dim, D, H, W)
        return x


class Coformer3D(nn.Module):

    def __init__(self,
                 patch_size=(1, 4, 4),
                 in_chans=1,
                 scale_factor=None,
                 embed_dim=(32, 64, 128, 256),
                 depths=[2, 2, 2, 2],
                 num_heads=[3, 6, 12, 12],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=LayerNorm,
                 act_layer=nn.GELU,
                 patch_norm=False,
                 frozen_stages=-1,
                 out_indices=(0, 1, 2, 3),
                 ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.patch_size = patch_size
        # self.position_embed = build_position_encoding(mode='v2', hidden_dim=embed_dim)
        self.scale_factor = scale_factor

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim[0],
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=embed_dim[i_layer],
                out_dim=embed_dim[i_layer + 1] if i_layer < self.num_layers - 1 else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                act_layer=act_layer,
                scale_factor=self.scale_factor[i_layer],
                downsample=Subsample if i_layer < self.num_layers - 1 else None,
            )
            self.layers.append(layer)

        # num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = embed_dim

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        outs = []
        for i, layer in enumerate(self.layers):
            out_, x = layer(x.contiguous())
            outs.append(out_)

        return tuple(outs)


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


if __name__ == '__main__':
    print("Hello")
    x = torch.randn(4, 1, 16, 320, 320).cuda()
    print(x.shape)
    scale_factor_ = [(1, 0.5, 0.5), (1, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
    encoder = Metaformer3D(patch_size=(1, 4, 4), scale_factor=scale_factor_).cuda()
    out = encoder(x)
    print(out[0].shape)
