import warnings

from .backbones import *  # noqa: F401,F403
from .builder import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                      ROI_EXTRACTORS, SHARED_HEADS, build_backbone,
                      build_detector, build_head, build_loss, build_neck,
                      build_roi_extractor, build_shared_head)
try:
    from .dense_heads import *  # noqa: F401,F403
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .dense_heads ({e})')
try:
    from .detectors import *  # noqa: F401,F403
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .detectors ({e})')
try:
    from .losses import *  # noqa: F401,F403
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .losses ({e})')
from .necks import *  # noqa: F401,F403
try:
    from .roi_heads import *  # noqa: F401,F403
except Exception as e:  # e.g. needs mmcv-full ops; not used by DSERT
    warnings.warn(f'mmdet: skipped .roi_heads ({e})')

__all__ = [
    'BACKBONES', 'NECKS', 'ROI_EXTRACTORS', 'SHARED_HEADS', 'HEADS', 'LOSSES',
    'DETECTORS', 'build_backbone', 'build_neck', 'build_roi_extractor',
    'build_shared_head', 'build_head', 'build_loss', 'build_detector'
]
__all__ = [name for name in __all__ if name in globals()]  # drop skipped modules
