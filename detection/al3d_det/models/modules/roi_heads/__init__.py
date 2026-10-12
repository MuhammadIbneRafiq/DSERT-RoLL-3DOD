from .LoGoHead import LoGoHead
__all__ = {
    'LoGoHead': LoGoHead,
}
try:
    # Imports the camera fusion modules at load time, so it is optional in LiDAR-only environments.
    from .LoGoHead_kitti import LoGoHeadKITTI
    __all__['LoGoHeadKITTI'] = LoGoHeadKITTI
except ImportError:
    pass