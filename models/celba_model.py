from typing import Tuple, Union
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch import Tensor

class DownConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 4,
        stride: int = 2,
        padding: int = 1,
        activation: str = "leaky_relu",
        do_batch_norm: bool = True,
        num_groups: int = 32,
        negative_slope: float = 0.2,
    ):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups, in_channels) if do_batch_norm else None
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel, stride=stride, padding=padding
        )

        match activation.lower():
            case "leaky_relu":
                self.act_fn = nn.LeakyReLU(negative_slope=negative_slope)
            case "identity":
                self.act_fn = nn.Identity()
            case _:
                raise NotImplementedError("`activation` must be `leaky_relu` or `identity`")

    def forward(self, x: Tensor) -> Tensor:
        if self.norm:
            x = self.norm(x)
        x = self.conv(x)
        x = self.act_fn(x)
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        down_out_channels: Tuple[int],
        kernels: Union[int, Tuple[int]],
        strides: Union[int, Tuple[int]],
        paddings: Union[int, Tuple[int]],
        do_batch_norms: Union[bool, Tuple[bool]],
        activations: Union[str, Tuple[str]],
    ):
        super().__init__()

        # check inputs
        num_blocks = len(down_out_channels)
        if not isinstance(kernels, int) and len(kernels) != num_blocks:
            raise ValueError("`kernels` must have the same length as `down_out_channels`")
        if not isinstance(strides, int) and len(strides) != num_blocks:
            raise ValueError("`strides` must have the same length as `down_out_channels`")
        if not isinstance(paddings, int) and len(paddings) != num_blocks:
            raise ValueError("`paddings` must have the same length as `down_out_channels`")
        if not isinstance(do_batch_norms, bool) and len(do_batch_norms) != num_blocks:
            raise ValueError("`do_batch_norms` must have the same length as `down_out_channels`")
        if not isinstance(activations, str) and len(activations) != num_blocks:
            raise ValueError("`activations` must have the same length as `down_out_channels`")
        
        if isinstance(kernels, int):
            kernels = (kernels,) * num_blocks
        if isinstance(strides, int):
            strides = (strides,) * num_blocks
        if isinstance(paddings, int):
            paddings = (paddings,) * num_blocks
        if isinstance(do_batch_norms, bool):
            do_batch_norms = (do_batch_norms,) * num_blocks
        if isinstance(activations, str):
            activations = (activations,) * num_blocks
        
        self.down_blocks = nn.Sequential()
        for i in range(num_blocks):
            out_channels = down_out_channels[i]
            self.down_blocks.append(
                DownConv(
                    in_channels,
                    out_channels,
                    kernel=kernels[i],
                    stride=strides[i],
                    padding=paddings[i],
                    activation=activations[i],
                    do_batch_norm=do_batch_norms[i],
                )
            )
            in_channels = out_channels
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.down_blocks(x)
        return x

class UpConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 4,
        stride: int = 2,
        padding: int = 1,
        activation: str = "relu",
        do_batch_norm: bool = True,
        num_groups: int = 32,
    ):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups, in_channels) if do_batch_norm else None
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel, stride=stride, padding=padding
        )

        match activation.lower():
            case "relu":
                self.act_fn = nn.ReLU()
            case "tanh":
                self.act_fn = nn.Tanh()
            case _:
                raise NotImplementedError("`activation` must be `relu` or `tanh`")

    def forward(self, x: Tensor) -> Tensor:
        if self.norm:
            x = self.norm(x)
        x = self.conv(x)
        x = self.act_fn(x)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        up_out_channels: Tuple[int],
        kernels: Union[int, Tuple[int]],
        strides: Union[int, Tuple[int]],
        paddings: Union[int, Tuple[int]],
        do_batch_norms: Union[bool, Tuple[bool]],
        activations: Union[str, Tuple[str]],
    ):
        super().__init__()

        # check inputs
        num_blocks = len(up_out_channels)
        if not isinstance(kernels, int) and len(kernels) != num_blocks:
            raise ValueError("`kernels` must have the same length as `up_out_channels`")
        if not isinstance(strides, int) and len(strides) != num_blocks:
            raise ValueError("`strides` must have the same length as `up_out_channels`")
        if not isinstance(paddings, int) and len(paddings) != num_blocks:
            raise ValueError("`paddings` must have the same length as `up_out_channels`")
        if not isinstance(do_batch_norms, bool) and len(do_batch_norms) != num_blocks:
            raise ValueError("`do_batch_norms` must have the same length as `up_out_channels`")
        if not isinstance(activations, str) and len(activations) != num_blocks:
            raise ValueError("`activations` must have the same length as `up_out_channels`")
        
        if isinstance(kernels, int):
            kernels = (kernels,) * num_blocks
        if isinstance(strides, int):
            strides = (strides,) * num_blocks
        if isinstance(paddings, int):
            paddings = (paddings,) * num_blocks
        if isinstance(do_batch_norms, bool):
            do_batch_norms = (do_batch_norms,) * num_blocks
        if isinstance(activations, str):
            activations = (activations,) * num_blocks
        
        self.up_blocks = nn.Sequential()
        for i in range(num_blocks):
            out_channels = up_out_channels[i]
            self.up_blocks.append(
                UpConv(
                    in_channels,
                    out_channels,
                    kernel=kernels[i],
                    stride=strides[i],
                    padding=paddings[i],
                    activation=activations[i],
                    do_batch_norm=do_batch_norms[i],
                )
            )
            in_channels = out_channels
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.up_blocks(x)
        return x

@dataclass
class EncoderConfig:
    in_channels: int = 3
    down_out_channels: Tuple[int] = (64, 128, 256, 512, 512)
    kernels: Union[int, Tuple[int]] = 4
    strides: Union[int, Tuple[int]] = (2, 2, 2, 2, 1)
    paddings: Union[int, Tuple[int]] = (1, 1, 1, 1, 0)
    do_batch_norms: Union[bool, Tuple[bool]] = (False, True, True, True, False)
    activations: Union[str, Tuple[str]] = (
        "leaky_relu",
        "leaky_relu",
        "leaky_relu",
        "leaky_relu",
        "identity",
    )

@dataclass
class DecoderConfig:
    in_channels: int = 512
    up_out_channels: Tuple[int] = (512, 256, 128, 64, 3)
    kernels: Union[int, Tuple[int]] = 4
    strides: Union[int, Tuple[int]] = (1, 2, 2, 2, 2)
    paddings: Union[int, Tuple[int]] = (0, 1, 1, 1, 1)
    do_batch_norms: Union[bool, Tuple[bool]] = (True, True, True, True, False)
    activations: Union[str, Tuple[str]] = (
        "relu",
        "relu",
        "relu",
        "relu",
        "tanh",
    )

class CelebAModel(nn.Module):
    def __init__(self, encoder_config: EncoderConfig, decoder_config: DecoderConfig):
        super().__init__()

        if decoder_config.in_channels != encoder_config.down_out_channels[-1]:
            raise ValueError(
                "`in_channels` for decoder must be the same as the last element of `down_out_channels` for encoder"
            )

        self.encoder = Encoder(**asdict(encoder_config))
        self.decoder = Decoder(**asdict(decoder_config))
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
