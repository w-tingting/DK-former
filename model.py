import torch
import torch.nn as nn
import math

from utils import KL_diagLog
from likelihood import Softmax
from layers import Conv_RFF, Conv_Linear, FullyConnected, Resnet, ConvBlock

in_channels = [0]


def _5d_to_3d(x, use_reshape=False):
    n, mc, C, H, W = x.shape
    if use_reshape:
        x = x.reshape(n, mc * C, H, W)
        x = x.reshape(n, mc * C, H * W)
    else:
        x = x.view(n, mc * C, H, W)
        x = x.view(n, mc * C, H * W)
    return x


class RFF_Token_Embed(nn.Module):
    def __init__(self, mc, kernel_type, block_conf, image_size=56, stage=1):
        super(RFF_Token_Embed, self).__init__()
        self.mc = mc
        self.kernel_type = kernel_type
        self.block_conf = block_conf
        self.stage = stage

        self.res_normal = Resnet()
        # self.res_down = Resnet(stride=2)

        self.blocks = nn.ModuleList()
        self.res = nn.ModuleList()

        # in_channels = [0]
        for i, bc in enumerate(block_conf):

            padding = bc[1] // 2
            self.grid_size = torch.tensor((image_size - bc[1] + 2 * padding) / bc[2] + 1).floor()
            image_size = self.grid_size

            if bc[2] >= 2:
                res_down = Resnet(k_size=bc[1], stride=bc[2])
            else:
                res_down = self.res_normal
            self.res.append(res_down)

            if stage == 1:
                b = ConvBlock(
                    in_channels=3 if i == 0 else in_channels[-2] + in_channels[-1],
                    out_channels=bc[0],
                    k_size=bc[1],
                    stride=bc[2],
                    mc=self.mc,
                    kernel_type=self.kernel_type,
                    group=1,
                    head=i == 0
                )
            else:
               
                b = ConvBlock(
                    in_channels=in_channels[-3] + in_channels[-2] + in_channels[-1] if i == 1 else in_channels[-2] +
                                                                                                   in_channels[-1],
                    out_channels=bc[0],
                    k_size=bc[1],
                    stride=bc[2],
                    mc=self.mc,
                    kernel_type=self.kernel_type,
                    group=1,
                    head=False
                )

            self.blocks.append(b)
            in_channels.append(bc[0])

        if stage == 1:
          
            self.layer_norm = torch.nn.LayerNorm((in_channels[-2] + in_channels[-1]) * self.mc, eps=1e-5)
        else:
            self.layer_norm = torch.nn.LayerNorm((in_channels[-2] + in_channels[-1]) * self.mc, eps=1e-5)
        # EXCEPT FULLY RESNET
        # in_channels.append(0)

        # num_pathces
        self.num_patches = self.grid_size * self.grid_size

    def forward(self, x):
        # x : [n, mc, c, h, w]
        layer_results = []
        # New Resnet
        if self.stage != 1:
            layer_results.append(x)

        for b in self.blocks:
            x = b(x)

            layer_results.append(x)
            if self.stage == 1:
                if len(layer_results) >= 2:
                    x = self.res[len(layer_results) - 1](layer_results[-2], x)
                    
            else:
                x = self.res[len(layer_results) - 2](layer_results[-2], x)

        # print("RFF_Token_x:", x.shape)
        bs, mc, C, H, W = x.shape
        # x = x.flatten(3)
        x = _5d_to_3d(x).transpose(1, 2)
        # Layer Norm
        x = self.layer_norm(x)
        return x, bs, mc, C, H, W


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 mc=5,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 is_final=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.final = is_final

        in_channels = int(dim / mc)

        self.conv_proj_q = ConvBlock(in_channels, in_channels, k_size=3, stride=1, mc=mc, kernel_type="arccos", group=1,
                                     head=0)
        self.conv_proj_k = ConvBlock(in_channels, in_channels, k_size=3, stride=1, mc=mc, kernel_type="arccos", group=1,
                                     head=0)
        self.conv_proj_v = ConvBlock(in_channels, in_channels, k_size=3, stride=1, mc=mc, kernel_type="arccos", group=1,
                                     head=0)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        # self.proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(in_channels, in_channels)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # print("x:",x.shape)
        if self.final == True:
            # bs, mc, cls, C
            cls = x[:, :, 0, :]
            x = x[:, :, 1:, :]
            bs, mc, F, C = x.shape
            H = W = int(math.sqrt(F))
            x = x.reshape(bs, mc, H, W, C).permute(0, 1, 4, 2, 3)
        else:
            bs, mc, C, H, W = x.shape

        q, k, v = self.conv_proj_q(x), self.conv_proj_k(x), self.conv_proj_v(x)
        if self.final == True:
            cls = cls.repeat(1, 1, 1, 1).permute(1, 2, 0, 3)
            q = q.reshape(bs, mc, C, H * W).permute(0, 1, 3, 2)
            k = k.reshape(bs, mc, C, H * W).permute(0, 1, 3, 2)
            v = v.reshape(bs, mc, C, H * W).permute(0, 1, 3, 2)
            q, k, v = torch.cat([cls, q], dim=2), torch.cat([cls, k], dim=2), torch.cat([cls, v], dim=2)

        # [batch_size, mc, num_heads, num_patches + 1, embed_dim_per_head]
        q = q.reshape(bs, mc, self.num_heads, C // self.num_heads, -1).permute(0, 1, 2, 4, 3)
        k = k.reshape(bs, mc, self.num_heads, C // self.num_heads, -1).permute(0, 1, 2, 4, 3)
        v = v.reshape(bs, mc, self.num_heads, C // self.num_heads, -1).permute(0, 1, 2, 4, 3)

        # print("q:", q.shape)
        # print("k:", k.shape)
        # print("v:", v.shape)

        # transpose: -> [batch_size, mc, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, mc, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # print("attn:", attn.shape)

        # @: multiply -> [batch_size, mc, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, mc, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        # x = (attn @ v).permute(0, 3, 1, 2, 4).reshape(bs, H * W, mc * C)
        x = (attn @ v).permute(0, 3, 1, 2, 4).reshape(bs, -1, mc * C)

        # bs*mc, HW, C
        x = x.reshape(bs, -1, mc, C).permute(0, 2, 1, 3)
        x = x.reshape(bs * mc, -1, C)
        # nn.Linear
        x = self.proj(x)
        # bs, HW, mc*C
        x = x.reshape(bs, mc, -1, C).permute(0, 2, 1, 3)
        x = x.reshape(bs, -1, mc * C)
        # nn.Dropout
        x = self.proj_drop(x)
      
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mc,
                 mlp_ratio=4.,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 is_final=False
                 ):
        super(Block, self).__init__()
        # self.layer_norm = torch.nn.LayerNorm(dim, eps=1e-5)

        self.final = is_final
        self.attn = Attention(dim, num_heads=num_heads, mc=mc, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio, is_final=is_final)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
       
        in_features = int(dim / mc)
        mlp_hidden_dim = int(in_features * mlp_ratio)
        
        self.mlp = Mlp(dim, mc, in_features=in_features, hidden_features=mlp_hidden_dim, drop=drop_ratio)

    def forward(self, x):

        res = x
        mhsa = self.drop_path(self.attn(x))
        # mhsa = self.attn(x)

        if self.final == True:
            bs, mc, N, C = res.shape
            # N = 17
            # H = W= int(math.sqrt(N))
            res = res.permute(0, 2, 1, 3).reshape(bs, N, mc * C)
            mhsa = res + mhsa
            # print("mhsa:", mhsa.shape)
        else:
            bs, mc, C, H, W = x.shape
            # bs, mc * C, H * W
            x = _5d_to_3d(x)
            x = x.permute(0, 2, 1)
            mhsa = x + mhsa

        out = self.drop_path(self.mlp(mhsa))
        # out = self.mlp(mhsa)
        out = mhsa + out
        # print("out:", out.shape)
        if self.final == True:
            out = out.reshape(bs, mc, C, -1).permute(0, 1, 3, 2)
        else:
            out = out.permute(0, 2, 1).reshape(bs, mc, C, H, W)
        
        return out


def drop_path(x, drop_prob: float = 0., training: bool = False):
   
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, dim, mc, in_features=None, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.layer_norm = nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        # self.drop = nn.Dropout(drop)
        self.mc = mc

    def forward(self, x):
        # bs, HW, mc*C
        bs, HW, C_mc = x.shape
        x = self.layer_norm(x)
        # bs*mc, HW, C
        x = x.reshape(bs, HW, self.mc, -1).permute(0, 2, 1, 3)
        x = x.reshape(bs * self.mc, HW, -1)
        # mlp
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        # bs, HW, mc*C
        x = x.reshape(bs, self.mc, HW, -1).permute(0, 2, 1, 3)
        x = x.reshape(bs, HW, -1)

        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class RFFTransformer(nn.Module):
    def __init__(self, num_classes=10, emb_dim_list=[16, 48, 128], block_list=[1, 2, 1], num_heads=8, mlp_ratio=4.0,
                 mc=5, kernel_type="arccos", block_conf=None, qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0.):
        super(RFFTransformer, self).__init__()
        self.num_classes = num_classes
        self.mc = mc
        self.kernel_type = kernel_type
        self.block_conf = block_conf

        # stage_1
        # self.rff_emb_1: [n, h * w, mc * c]
        self.rff_emb_1 = RFF_Token_Embed(self.mc, self.kernel_type, self.block_conf[0], stage=1)

        # MHSA
        # stage_1
        dpr_1 = [x.item() for x in torch.linspace(0, drop_path_ratio, block_list[0])]  # stochastic depth decay rule

        self.blocks_1 = nn.Sequential(*[
            Block(dim=128 * self.mc, num_heads=num_heads, mc=self.mc, mlp_ratio=mlp_ratio,
                  qk_scale=qk_scale, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr_1[i],
                  is_final=False)
            for i in range(block_list[0])])

        # head: change value
        last_channel = 64
        self.head = Conv_RFF(128, last_channel, 3, 1, 1, self.mc, self.kernel_type, 1, False)
        self.fully = FullyConnected(1 * 1 * last_channel, self.num_classes, self.mc, False)


    def compute_objective(self, y_pred, y, num):
        ## Given the output layer, we compute the conditional likelihood across all samples
        softmax = Softmax(self.mc, self.num_classes)
        ll = softmax.log_cond_prob(y, y_pred)
        ell = torch.sum(torch.mean(ll, 0)) * num
        return ell

    def get_kl(self):
        kl = 0
        for mod in self.modules():
            if isinstance(mod, Conv_RFF):
                # print(mod.__str__())
                Omega_mean_prior, Omega_logsigma_prior = mod.get_prior_Omega()
                kl += KL_diagLog(mod.Omega_mean, Omega_mean_prior, mod.Omega_logsigma,
                                 Omega_logsigma_prior)
            elif isinstance(mod, (Conv_Linear, FullyConnected)):
                # print(mod.__str__())
                # print(mod.W_eps[0])
                kl += KL_diagLog(mod.W_mean, mod.W_mean_prior, mod.W_logsigma,
                                 mod.W_logsigma_prior)
        return kl

    def forward(self, x):
        x, bs, mc, C, H, W = self.rff_emb_1(x)  # (b_s, h*w,mc*2*c)
       
        # # cls_token
        # # bs h*w mc*c -> bs mc (h w ) c
        x = x.reshape(bs, H * W, mc, C).permute(0, 2, 1, 3)
        cls = self.class_token.repeat(x.shape[0], 1, 1, 1)
        
        x = torch.cat([cls, x], dim=2)
        # ********************************************************************************************************
        # bs mc (cls_token+h*w) c
        x = self.blocks_1(x)
        # ********************************************************************************************************
        # # cls_token
        # 1,5,17,128
        x = x[:, :, 0, :].unsqueeze(2)
        bs, mc, N, C = x.shape
        x = x.reshape(bs, mc, 1, 1, C).permute(0, 1, 4, 2, 3)
        x = self.head(x)  # rff
        x = x.reshape(bs, self.mc, -1)
        x = x.transpose(0, 1)
        x = self.fully(self.layer_norm(x))
        return x
       