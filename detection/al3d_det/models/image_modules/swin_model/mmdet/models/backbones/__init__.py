import warnings

try:
    from .darknet import Darknet
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .darknet ({e})')
try:
    from .detectors_resnet import DetectoRS_ResNet
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .detectors_resnet ({e})')
try:
    from .detectors_resnext import DetectoRS_ResNeXt
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .detectors_resnext ({e})')
try:
    from .hourglass import HourglassNet
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .hourglass ({e})')
try:
    from .hrnet import HRNet
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .hrnet ({e})')
try:
    from .regnet import RegNet
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .regnet ({e})')
try:
    from .res2net import Res2Net
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .res2net ({e})')
try:
    from .resnest import ResNeSt
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .resnest ({e})')
try:
    from .resnet import ResNet, ResNetV1d
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .resnet ({e})')
try:
    from .resnext import ResNeXt
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .resnext ({e})')
try:
    from .ssd_vgg import SSDVGG
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .ssd_vgg ({e})')
try:
    from .trident_resnet import TridentResNet
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .trident_resnet ({e})')
from .swin_transformer import SwinTransformer
try:
    from .cbnet import CBSwinTransformer
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .cbnet ({e})')
__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'Res2Net',
    'HourglassNet', 'DetectoRS_ResNet', 'DetectoRS_ResNeXt', 'Darknet',
    'ResNeSt', 'TridentResNet', 'SwinTransformer', 'CBSwinTransformer'
]
__all__ = [name for name in __all__ if name in globals()]  # drop skipped modules
