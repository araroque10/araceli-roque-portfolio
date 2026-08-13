import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class ModulatedConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate

        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )
        self.style = nn.Linear(style_dim, in_channels)

    def forward(self, x, style):
        batch, _, D, H, W = x.shape
       
        style = self.style(style)  # Proyectamos al espacio correcto
        assert style.shape == (batch, self.in_channels), (
            f"Shape incorrecto: style.shape={style.shape}, esperado=({batch}, {self.in_channels})"
        )

        style = style.view(batch, 1, self.in_channels, 1, 1, 1)
        weight = self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4, 5]) + self.eps)
            weight = weight * demod.view(batch, self.out_channels, 1, 1, 1, 1)

        x = x.view(1, -1, D, H, W)
        weight = weight.view(batch * self.out_channels, self.in_channels,
                             self.kernel_size, self.kernel_size, self.kernel_size)
        out = F.conv3d(x, weight, padding=self.kernel_size // 2, groups=batch)
        return out.view(batch, self.out_channels, D, H, W)

class StyledConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, kernel_size=3):
        super().__init__()
        self.conv = ModulatedConv3d(in_channels, out_channels, kernel_size, style_dim)
        self.noise_strength = nn.Parameter(torch.zeros(1))
        self.activation = nn.LeakyReLU(0.2, inplace=False)

        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1, 1))

    def forward(self, x, style, noise=None):
        out = self.conv(x, style)
        if noise is not None:
            out = out + self.noise_strength * noise
        out = out + self.bias 
        out = self.activation(out) 

        return out

def make_mapping_layers(size_in, num_layers=8):
    layers = []
    for _ in range(num_layers):
        layers.append(nn.Linear(size_in, size_in))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.LayerNorm(size_in)) 
    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, output_shape=32, size_in=256, nch=96, depth=2, ws=True, activ=torch.nn.LeakyReLU(0.2)):
        super().__init__()
        self.size_in = size_in
        self.mapping = nn.Sequential(
            *make_mapping_layers(size_in),
            nn.Linear(size_in, sum([max(nch // (2 ** i), 16) for i in range(int(torch.log2(torch.tensor(output_shape // 8)).item()))]))
        )

        self.initial_const = nn.Parameter(torch.randn(1, nch, 8, 8, 8))
        self.blocks = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        resolution = output_shape
        self.style_dims = []

        steps = int(torch.log2(torch.tensor(resolution // 8)).item())
        channels = nch
        self.upsample_layers = nn.ModuleList()
        self.conv_after_up_layers = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        resolution = output_shape
        self.style_dims = []

        steps = int(torch.log2(torch.tensor(resolution // 8)).item())

        channels = nch

        self.upsample_layers = nn.ModuleList()
        self.conv_after_up_layers = nn.ModuleList()

        for _ in range(steps):
            next_channels = max(channels // 2, 16)

            self.upsample_layers.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False))
            self.conv_after_up_layers.append(nn.Conv3d(channels, next_channels, kernel_size=3, padding=1))

            self.style_dims.append(next_channels)
            self.blocks.append(StyledConvBlock(next_channels, next_channels, style_dim=next_channels))
            self.to_rgbs.append(nn.Conv3d(next_channels, 1, kernel_size=1))
            channels = next_channels

        self.to_rgb = nn.Conv3d(channels, 1, kernel_size=1)

    def forward(self, z):
        batch_size = z.size(0)

        styles_split = []
        for dim in self.style_dims:
            if torch.rand(1).item() < 0.9:
                z_perm = z[torch.randperm(z.size(0))]
                mapped = self.mapping(z_perm)
            else:
                mapped = self.mapping(z)
            styles_split.append(mapped[:, :dim])

        x = self.initial_const.expand(batch_size, -1, -1, -1, -1)
        rgb = None
        for idx, (upsample, conv_after_up, block) in enumerate(zip(self.upsample_layers, self.conv_after_up_layers, self.blocks)):
            style = styles_split[idx]
            x = upsample(x)
            x = conv_after_up(x)
            x = block(x, style)

            rgb_block = self.to_rgbs[idx](x)
            if rgb is None:
                rgb = rgb_block
            else:
                rgb = F.interpolate(rgb, size=rgb_block.shape[2:], mode='trilinear', align_corners=False)
                rgb = rgb + rgb_block

        return torch.sigmoid(rgb)


class MinibatchStdDev(nn.Module):
    def forward(self, x):
        batch, _, D, H, W = x.shape
        std = torch.std(x, dim=0, keepdim=True) 
        mean_std = std.mean()       
        shape = (batch, 1, D, H, W)
        std_feature = mean_std.expand(shape)    
        return torch.cat([x, std_feature], 1)  

class Discriminator(nn.Module):
    def __init__(self, output_shape=32, nch=64, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        resolution = output_shape
        steps = int(torch.log2(torch.tensor(resolution // 4)).item())

        self.blocks.append(nn.Conv3d(1, nch, 3, padding=1))
        channels = nch

        for _ in range(1, steps):
            next_channels = channels * 2
            self.blocks.append(nn.Conv3d(channels, next_channels, 3, padding=1))
            channels = next_channels

        self.stddev = MinibatchStdDev()
        self.final_conv = nn.Conv3d(channels + 1, 1, kernel_size=4)

    def forward(self, x):
        for block in self.blocks:
            x = F.leaky_relu(block(x), 0.2, inplace=False)
            x = F.avg_pool3d(x, 2)

        x = self.stddev(x)           
        x = self.final_conv(x)     
        return x.view(x.size(0), -1)


def ceil(x):
    return int(x - int(x+1)) + int(x+1)