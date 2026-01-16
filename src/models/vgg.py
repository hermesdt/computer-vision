from typing_extensions import Self
import torch
from torch import nn
from typing import List, Optional, Type, Dict, Any, Literal, overload

_ConfigType = list[int | str]
_cfgs: Dict[str, _ConfigType] = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class ConvMlp(nn.Module):
    """Convolutional MLP block for VGG head.

    Replaces traditional Linear layers with Conv2d layers in the classifier.
    """

    def __init__(
            self,
            in_features: int = 512,
            out_features: int = 4096,
            kernel_size: int = 7,
            mlp_ratio: float = 1.0,
            drop_rate: float = 0.2,
            act_layer: Type[nn.Module] = nn.ReLU,
            conv_layer: Type[nn.Module] = nn.Conv2d,
            device=None,
            dtype=None,
    ) -> None:
        """Initialize ConvMlp.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            kernel_size: Kernel size for first conv layer.
            mlp_ratio: Ratio for hidden layer size.
            drop_rate: Dropout rate.
            act_layer: Activation layer type.
            conv_layer: Convolution layer type.
        """
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.input_kernel_size = kernel_size
        mid_features = int(out_features * mlp_ratio)
        self.fc1 = conv_layer(in_features, mid_features, kernel_size, bias=True, **dd)
        self.act1 = act_layer(True)
        self.drop = nn.Dropout(drop_rate)
        self.fc2 = conv_layer(mid_features, out_features, 1, bias=True, **dd)
        self.act2 = act_layer(True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int = 512,
        mlp_ratio: float = 1.0,
        out_features: int = 4096,
        act_layer: Type[nn.Module] = nn.ReLU,
        drop_rate: float = 0.0,
    ) -> None:
        """Initialize MLP module.

        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden features.
            out_features: Number of output features.
            act_layer: Activation layer type.
            drop: Dropout rate.
        """
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = int(out_features * mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        return x

@overload
def _create_pre_head_layer(
    *,
    pre_head_layer: Literal['Mlp'] = 'Mlp',
    in_features: int,
    mlp_ratio: float,
    out_features: int,
    act_layer: Type[nn.Module],
    drop_rate: float = 0.0,
    **kwargs,
) -> Mlp:
    return Mlp(
        in_features=in_features,
        mlp_ratio=mlp_ratio,
        out_features=out_features,
        act_layer=act_layer,
        drop_rate=drop_rate,
    )

@overload
def _create_pre_head_layer(
    *,
    pre_head_layer: Literal['ConvMlp'] = 'ConvMlp',
    in_features: int,
    out_features: int,
    kernel_size: int,
    mlp_ratio: float,
    drop_rate: float,
    act_layer: Type[nn.Module],
    conv_layer: Type[nn.Module],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    **kwargs,
) -> ConvMlp:
    return ConvMlp(
        in_features=in_features,
        out_features=out_features,
        kernel_size=kernel_size,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        act_layer=act_layer,
        conv_layer=conv_layer,
        device=device,
        dtype=dtype,
        **kwargs,
    )

def _create_pre_head_layer(
    pre_head_layer: Literal['Mlp', 'ConvMlp'] = 'Mlp',
    **kwargs: Any,
) -> Mlp | ConvMlp:
    if pre_head_layer == 'Mlp':
        return Mlp(**kwargs)
    elif pre_head_layer == 'ConvMlp':
        return ConvMlp(**kwargs)
    else:
        raise ValueError(f"Unsupported pre_head_layer: {pre_head_layer}")

class VGG(nn.Module):
    def __init__(
        self,
        cfg: _ConfigType,
        num_classes: int = 1000,
        in_chans: int = 3,
        output_stride: int = 32,
        mlp_ratio: float = 1.0,
        act_layer: Type[nn.Module] = nn.ReLU,
        conv_layer: Type[nn.Module] = nn.Conv2d,
        pre_head_layer: Literal['Mlp', 'ConvMlp'] = 'ConvMlp',
        norm_layer: Optional[Type[nn.Module]] = None,
        global_pool: str = 'avg',
        drop_rate: float = 0,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize VGG model.

        Args:
            cfg: Configuration list defining network architecture.
            num_classes: Number of classes for classification.
            in_chans: Number of input channels.
            output_stride: Output stride of network.
            mlp_ratio: Ratio for MLP hidden layer size.
            act_layer: Activation layer type.
            conv_layer: Convolution layer type.
            norm_layer: Normalization layer type.
            global_pool: Global pooling type.
            drop_rate: Dropout rate.
        """

        super(VGG, self).__init__()
        dd: dict[str, Any] = {'device': device, 'dtype': dtype}
        assert output_stride == 32
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.feature_info = []

        prev_chs = in_chans
        net_stride = 1
        pool_layer = nn.MaxPool2d
        layers: List[nn.Module] = []

        for v in cfg:
            if v == 'M':
                layers.append(pool_layer(kernel_size=2, stride=2))
            else:
                v = int(v)
                conv = conv_layer(prev_chs, v, kernel_size=3, padding=1, stride=1, **dd)
                if norm_layer is not None:
                    layers.extend([conv, norm_layer(v), act_layer(inplace=True)])
                else:
                    layers.extend([conv, act_layer(inplace=True)])
                prev_chs = v
    
        self.features = nn.Sequential(*layers)
        self.pre_logits = _create_pre_head_layer(
            pre_head_layer=pre_head_layer,
            in_features=prev_chs,
            mlp_ratio=mlp_ratio,
            out_features=4096,
            act_layer=act_layer,
            kernel_size=7,
            conv_layer=conv_layer,
            drop_rate=drop_rate,
            **dd
        )


        

