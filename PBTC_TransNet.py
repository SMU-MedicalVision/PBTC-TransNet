"""
author: Chuanpu Li
date: 2023_09_21 10:28
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock2D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=(3, 3), stride=stride,
                               padding=(1, 1), bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1, 1), downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = nn.Conv3d(in_channels=inplanes, out_channels=planes, kernel_size=(3, 3, 3), stride=stride,
                               padding=(1, 1, 1), bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels=planes, out_channels=planes, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1), bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class Encoder2D(nn.Module):  # ref: ResNet
    def __init__(self, in_channels=1, basic_dims=32, norm_layer=None, num_layers=[2, 2, 2, 2]):
        super(Encoder2D, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=basic_dims, kernel_size=(7, 7), stride=(2, 2),
                               padding=(3, 3),
                               bias=False)
        self.bn1 = norm_layer(basic_dims)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(in_planes=basic_dims, out_planes=basic_dims * 2, num_blocks=num_layers[0],
                                       stride=(2, 2))
        self.layer2 = self._make_layer(in_planes=basic_dims * 2, out_planes=basic_dims * 4, num_blocks=num_layers[1],
                                       stride=(2, 2))
        self.layer3 = self._make_layer(in_planes=basic_dims * 4, out_planes=basic_dims * 8, num_blocks=num_layers[2],
                                       stride=(2, 2))
        self.layer4 = self._make_layer(in_planes=basic_dims * 8, out_planes=basic_dims * 16, num_blocks=num_layers[3],
                                       stride=(2, 2))

    def _make_layer(self, in_planes, out_planes, num_blocks, stride=(1, 1)):
        norm_layer = self._norm_layer
        downsample = None

        if max(stride) != 1:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=stride, bias=False),
                norm_layer(out_planes),
            )

        layers = []
        layers.append(BasicBlock2D(in_planes, out_planes, stride=stride, downsample=downsample, norm_layer=norm_layer))

        for _ in range(1, num_blocks):
            layers.append(BasicBlock2D(out_planes, out_planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class Encoder3D(nn.Module):  # ref: ResNet
    def __init__(self, in_channels=1, basic_dims=32, norm_layer=None, num_layers=[2, 2, 2, 2]):
        super(Encoder3D, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=basic_dims, kernel_size=(3, 3, 3),
                               stride=(1, 1, 1), padding=(1, 1, 1),
                               bias=True)
        self.bn1 = norm_layer(basic_dims)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(in_planes=basic_dims, out_planes=basic_dims * 2, num_blocks=num_layers[0],
                                       stride=(2, 2, 2))
        self.layer2 = self._make_layer(in_planes=basic_dims * 2, out_planes=basic_dims * 4, num_blocks=num_layers[1],
                                       stride=(2, 2, 2))
        self.layer3 = self._make_layer(in_planes=basic_dims * 4, out_planes=basic_dims * 8, num_blocks=num_layers[2],
                                       stride=(1, 2, 2))

    def _make_layer(self, in_planes, out_planes, num_blocks, stride=(1, 1, 1)):
        norm_layer = self._norm_layer
        downsample = None

        if max(stride) != 1:
            downsample = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 1), stride=stride, bias=False),
                norm_layer(out_planes),
            )

        layers = []
        layers.append(BasicBlock3D(in_planes, out_planes, stride=stride, downsample=downsample, norm_layer=norm_layer))

        for _ in range(1, num_blocks):
            layers.append(BasicBlock3D(out_planes, out_planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class SelfAttention(nn.Module):
    def __init__(
            self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)  # qkv
                .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1, n_levels=1, n_points=4):
        super(Transformer, self).__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),  #
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )

        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)

    def forward(self, x, pos):
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x


class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()

    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()  # batch size, modal number, channel,
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        x = y.view(B, -1, H, W, Z)
        return x


class MaskModal_tensor(nn.Module):
    def __init__(self):
        super(MaskModal_tensor, self).__init__()

    def forward(self, ct_tensor, mr_tensor, x_tensor, mask):
        ct = torch.zeros_like(ct_tensor)
        mr = torch.zeros_like(mr_tensor)
        x = torch.zeros_like(x_tensor)
        ct[mask[..., 0]] = ct_tensor[mask[..., 0]]
        mr[mask[..., 1]] = mr_tensor[mask[..., 1]]
        x[mask[..., 2]] = x_tensor[mask[..., 2]]

        return ct, mr, x

class Model(nn.Module):
    def __init__(self, num_modals=3, num_classes=3, basic_dim=32, transformer_basic_dim=512, num_heads=8, mlp_dim=4096,
                 inter_depth=1, ct_feature_size=(3, 12, 12), mr_feature_size=(3, 12, 12),
                 x_feature_size=(16, 12), dropout_rate=0.2, clinical_dims=13):
        super(Model, self).__init__()

        self.transformer_basic_dim = transformer_basic_dim

        self.ct_encoder = Encoder3D(in_channels=1, basic_dims=basic_dim)
        self.mr_encoder = Encoder3D(in_channels=2, basic_dims=basic_dim)
        self.x_encoder = Encoder2D(in_channels=1, basic_dims=basic_dim)

        self.ct_encode_conv = nn.Conv3d(basic_dim * 8, transformer_basic_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                                        padding=0)
        self.mr_encode_conv = nn.Conv3d(basic_dim * 8, transformer_basic_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                                        padding=0)
        self.x_encode_conv = nn.Conv2d(basic_dim * 16, transformer_basic_dim, kernel_size=(1, 1), stride=(1, 1),
                                       padding=0)

        self.ct_pos = nn.Parameter(
            torch.zeros(1, ct_feature_size[0] * ct_feature_size[1] * ct_feature_size[2], transformer_basic_dim))
        self.mr_pos = nn.Parameter(
            torch.zeros(1, mr_feature_size[0] * mr_feature_size[1] * mr_feature_size[2], transformer_basic_dim))
        self.x_pos = nn.Parameter(torch.zeros(1, x_feature_size[0] * x_feature_size[1], transformer_basic_dim))

        ##### mask
        self.masker_tensor = MaskModal_tensor()

        self.multimodal_transformer = Transformer(embedding_dim=transformer_basic_dim, depth=inter_depth,
                                                  heads=num_heads, mlp_dim=mlp_dim, n_levels=num_modals)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(dropout_rate)
        self.multimodal_fc = nn.Linear(transformer_basic_dim + clinical_dims, num_classes)

        self.is_training = False

        for m in self.modules():  # init
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, ct, mr, x, clinical, mask):

        # extract feature from different layers
        ct_features = self.ct_encoder(ct)
        mr_features = self.mr_encoder(mr)
        x_features = self.x_encoder(x)

        ct_tokens = self.ct_encode_conv(ct_features).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1,
                                                                                              self.transformer_basic_dim)
        mr_tokens = self.mr_encode_conv(mr_features).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1,
                                                                                              self.transformer_basic_dim)
        x_tokens = self.x_encode_conv(x_features).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1,
                                                                                        self.transformer_basic_dim)

        ct_intra, mr_intra, x_intra = self.masker_tensor(ct_tokens, mr_tokens, x_tokens, mask)
        multimodal_token = torch.cat((ct_intra, mr_intra, x_intra), dim=1)
        multimodal_pos = torch.cat((self.ct_pos, self.mr_pos, self.x_pos), dim=1)
        multimodal_inter_token_x5 = self.multimodal_transformer(multimodal_token, multimodal_pos).permute(0, 2, 1)
        multimodal_avg = self.avg_pool(multimodal_inter_token_x5)
        multimodal_avg = torch.flatten(multimodal_avg, 1)

        multimodal_avg = self.dropout(multimodal_avg)
        multimodal_info = torch.cat((multimodal_avg, clinical), dim=1)
        multimodal_preds = self.multimodal_fc(multimodal_info)
        multimodal_output = torch.softmax(multimodal_preds, dim=1)

        return multimodal_preds, multimodal_output


if __name__ == '__main__':
    model = Model()
    ct = torch.randn(size=(1, 1, 12, 96, 96))  # bs, channel, z, x, y   CT_roi
    mr = torch.randn(size=(1, 2, 12, 96, 96))  # bs, channel, z, x, y   MR_roi
    x = torch.zeros(size=(1, 1, 512, 384))    # missing data
    mask = torch.tensor(([True, True, False]))   # modality situation
    clinical_info = torch.randn(size=(1, 13))   # bs, clinical(sex, ......)
    result = model(ct, mr, x, clinical_info, mask)