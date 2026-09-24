import warnings

try:
    from .bfp import BFP
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .bfp ({e})')
try:
    from .channel_mapper import ChannelMapper
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .channel_mapper ({e})')
try:
    from .fpg import FPG
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .fpg ({e})')
from .fpn import FPN
try:
    from .fpn_carafe import FPN_CARAFE
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .fpn_carafe ({e})')
try:
    from .hrfpn import HRFPN
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .hrfpn ({e})')
try:
    from .nas_fpn import NASFPN
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .nas_fpn ({e})')
try:
    from .nasfcos_fpn import NASFCOS_FPN
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .nasfcos_fpn ({e})')
try:
    from .pafpn import PAFPN
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .pafpn ({e})')
try:
    from .rfp import RFP
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .rfp ({e})')
try:
    from .yolo_neck import YOLOV3Neck
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .yolo_neck ({e})')
# from .MuFPN import NoFpnSoftDownSample
try:
    from .fpnc import FPNC
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .fpnc ({e})')
__all__ = [
    'FPN', 'BFP', 'ChannelMapper', 'HRFPN', 'NASFPN', 'FPN_CARAFE', 'PAFPN',
    'NASFCOS_FPN', 'RFP', 'YOLOV3Neck', 'FPG', 'FPNC'
]
__all__ = [name for name in __all__ if name in globals()]  # drop skipped modules
